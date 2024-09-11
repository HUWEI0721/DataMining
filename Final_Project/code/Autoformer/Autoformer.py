import math
import torch
import torch.nn as nn

from Autoformer.Layers.AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from Autoformer.Layers.Autoformer_EncDec import EncoderLayer, my_Layernorm, DecoderLayer, Encoder, Decoder
from Autoformer.Layers.Embed import DataEmbedding_wo_pos


# 使用Autoformer进行4步长的预测
class AutoformerModel(nn.Module):
    def __init__(self, configs):
        super(AutoformerModel, self).__init__()
        self.seq_len = configs.seq_len  # 利用历史时间序列的时间长度，编码器输入的时间维度
        self.pred_len = configs.pred_len  # 预测未来时间序列的时间长度，解码器输出的时间维度
        self.label_len = configs.label_len  # 每一个时间步的维度

        self.output_attention = False  # 是否使用输出注意力

        # Decomp，传入参数均值滤波器的核大小
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # embedding操作，由于时间序列天然在时序上具有先后关系，因此这里embedding的作用更多的是为了调整维度
        self.enc_embedding = DataEmbedding_wo_pos(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder，采用的是多编码层堆叠
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        # 这里的第一个False表明是否使用mask机制。
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False, configs=configs),
                        configs.d_model, configs.n_heads),
                    # 编码过程中的特征维度设置
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    # 激活函数
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            # 时间序列通常采用Layernorm而不适用BN层
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Decoder也是采用多解码器堆叠
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 如同传统的Transformer结构，decoder的第一个attention需要mask，保证当前的位置的预测不能看到之前的内容
                    # 这个做法是来源于NLP中的作法，但是换成时序预测，实际上应该是不需要使用mask机制的。
                    # 而在后续的代码中可以看出，这里的attention模块实际上都没有使用mask机制。

                    # self-attention，输入全部来自于decoder自身
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    # cross-attention，输入一部分来自于decoder，另一部分来自于encoder的输出
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    # 任务要求的输出特征维度
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc表示编码器的输入
        # x_mark_enc表示x_enc中各个时间戳的先后关系
        # x_dec表示解码器的输入
        # x_mark_dec表示x_dec中各个时间戳的先后关系
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        # Trend部分：前半部分来自于时序拆解，后半部分即预测部分用均值占位
        # Season部分：前半部分来自于时序拆解，后半部分即预测部分用零值占位
        dec_out = trend_part + seasonal_part
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


# 时序拆解器
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # 这里的判断语句主要是为了和Fedformer的部分进行区分，如果只考虑autoformer可以默认这里的判断全是True
        if isinstance(self.kernel_size, list):
            if len(self.kernel_size) == 1:
                self.kernel_size = self.kernel_size[0]
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
