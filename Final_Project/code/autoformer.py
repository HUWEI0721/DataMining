import numpy as np
import pandas as pd
import os


def createXY(dataset, npast, n_future, column_target):
    dataX, dataY = [], []
    for i in range(npast, dataset.shape[0] - n_future + 1):
        dataX.append(dataset.iloc[i - npast:i].values)
        dataY.append(dataset.iloc[i:i + n_future][column_target].values)
    return np.array(dataX), np.array(dataY)


def processFiles(need_columns, target_column, all_df, n_past=1, n_future=1):
    all_x, all_y = np.array([]), np.array([])
    for df in all_df:
        df = df[need_columns]
        x, y = createXY(df, n_past, n_future, target_column)
        all_x = np.vstack([all_x, x]) if all_x.size else x
        all_y = np.vstack([all_y, y]) if all_y.size else y
    return all_x, all_y


def processSummary(summary_columns, summary, folder_path):
    all_df = []
    for f in os.listdir(folder_path):
        if f.endswith('.xlsx') or f.endswith('.xls'):
            df = pd.read_excel(folder_path + f)
            df['Patient Number'] = f.split('.')[0]
            df = pd.merge(df, summary[summary_columns], on='Patient Number')

            all_df.append(df)
    return all_df


need_columns = ['Date', 'CGM (mg / dl)', 'CBG (mg / dl)']
target_column = 'CGM (mg / dl)'
summary_columns = ['Patient Number',
                   'Gender (Female=1, Male=2)',
                   'Age (years)',
                   'BMI (kg/m2)',
                   'Type of Diabetes',
                   'Alcohol Drinking History (drinker/non-drinker)']
need_columns.extend(summary_columns)

summary1 = pd.read_excel('./data/Shanghai_T1DM_Summary.xlsx')
summary2 = pd.read_excel('./data/Shanghai_T2DM_Summary.xlsx')

summary1['Type of Diabetes'] = summary1['Type of Diabetes'].apply(lambda x: 1 if x == 'T1DM' else 2)
summary2['Type of Diabetes'] = summary2['Type of Diabetes'].apply(lambda x: 1 if x == 'T1DM' else 2)

summary1['Alcohol Drinking History (drinker/non-drinker)'] = summary1[
    'Alcohol Drinking History (drinker/non-drinker)'].apply(lambda x: 1 if x == 'drinker' else 0)
summary2['Alcohol Drinking History (drinker/non-drinker)'] = summary2[
    'Alcohol Drinking History (drinker/non-drinker)'].apply(lambda x: 1 if x == 'drinker' else 0)

T1df_all = processSummary(summary_columns, summary1, './data/Shanghai_T1DM/')
T2df_all = processSummary(summary_columns, summary2, './data/Shanghai_T2DM/')

PAST_HOURS = 4  # 使用过去4小时的数据输入
FUTURE_HOURS = 1  # 预测未来1小时的数据
NPAST, NFUTURE = 4 * PAST_HOURS, 4 * FUTURE_HOURS
all_x, all_y = processFiles(need_columns, target_column, T1df_all + T2df_all, n_past=NPAST, n_future=NFUTURE)


# 把all_x的每一个元素都转换为一个dataframe，表头为need_columns
def to_df(x):
    return pd.DataFrame(x, columns=need_columns)


all_x_df = list(map(to_df, all_x))

# 删除Patient Number属性
for i in range(len(all_x_df)):
    all_x_df[i] = all_x_df[i].drop(columns=['Patient Number'])
need_columns.remove('Patient Number')

# 将Date转换为Hour，并附加到all_x_df的表尾
for i in range(len(all_x_df)):
    all_x_df[i]['Hour'] = all_x_df[i]['Date'].apply(lambda x: x.hour)
    # 使用正余弦函数将Hour转换为周期性特征
    all_x_df[i]['Hour_sin'] = np.sin(2 * np.pi * all_x_df[i]['Hour'] / 24)
    # 删除Hour列和Date列
    all_x_df[i] = all_x_df[i].drop(columns=['Date', 'Hour'])

need_columns.remove('Date')
need_columns.append('Hour_sin')

# 将all_x_df中的每个元素元素设置为float类型
all_x_df = [df.astype(float) for df in all_x_df]
all_x_df[0].head()

# 剩下的缺失值使用0填充
_all_x_df = [df.fillna(0) for df in all_x_df]
# 输出所有的_all_x_df中每个元素是否有缺失值的汇总
print(pd.Series(map(lambda x: x.isnull().sum().sum(), _all_x_df)).value_counts())

# 对于all_y，使用前一个值填充缺失值
_all_y = pd.DataFrame(all_y).fillna(method='ffill').values
# 输出all_y中的缺失值的个数
print(pd.Series(_all_y.flatten()).isnull().sum())

# TODO: 归一化
_all_x_df[0].head()

all_x_df = _all_x_df
all_y = _all_y

# 分割训练集与测试集
from sklearn.model_selection import train_test_split

SpiltRatio = 0.7
X_train, X_test, y_train, y_test = train_test_split(all_x_df, all_y, test_size=1 - SpiltRatio, random_state=42)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Autoformer.Autoformer import AutoformerModel as Autoformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:
    def __init__(self):
        self.d_feature = 1
        self.d_model = 128
        self.seq_len = 16
        self.pred_len = 4
        self.n_layers = 3
        self.n_heads = 8
        self.label_len = 1
        self.moving_avg = 4
        self.d_ff = 512
        self.dropout = 0.1
        self.activation = 'relu'
        self.factor = 4
        self.e_layers = 3
        self.d_layers = 3
        self.c_out = 1
        self.embed = 'timeF'
        self.freq = 'h'
        self.wavelet = 1


config = Config()
autoformer_model = Autoformer(config).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoformer_model.parameters(), lr=0.001)

# 训练模型
epochs = 10
batch_size = 64
train_loss = []
test_loss = []
for epoch in range(epochs):
    autoformer_model.train()
    for i in range(0, len(X_train), batch_size):
        x = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32).to(device)
        y = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        output = autoformer_model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    autoformer_model.eval()
    with torch.no_grad():
        x = torch.tensor(X_test, dtype=torch.float32).to(device)
        y = torch.tensor(y_test, dtype=torch.float32).to(device)

        output = autoformer_model(x)
        loss = criterion(output, y)
        test_loss.append(loss.item())

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss[-1]}, Test Loss: {test_loss[-1]}')
