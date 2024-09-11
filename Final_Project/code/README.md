# 软件版本
- CUDA：12.1
- Pytorch：2.3.0
- Python：3.12

# 环境配置
## 1. 安装相应的包
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install matplotlib
pip3 install numpy
pip3 install pandas
pip3 install scikit-learn
pip3 install pytorch_tcn
```

## 2. 配置数据集
确保数据集在`data`文件夹下，且其目录结构如下：
```
data
├── Shanghai_T1DM
│   ├── xxxx.csv
│   │
│   ├── xxxx.csv
│   │   
│   └── xxxx.csv
├── Shanghai_T2DM
│   ├── xxxx.csv
│   │
│   ├── xxxx.csv
│   │
│   └── xxxx.csv
├── Shanghai_T1DM_Summary.xlsx
│
└── Shanghai_T2DM_Summary.xlsx
```

# 运行
- tcn.ipynb, lstm.ipynb, gru.ipynb这三个jupyter notebook文件均包含了数据预处理与模型训练+预测的代码，可以直接运行。
- 以cat开头的jupyter notebook文件包含了模型特征重要性分析的代码，可以直接运行。
- Autoformer目录下包含了Autoformer模型的代码，尚不能直接运行。
- some_models 存放了一些模型的代码，由于受限于参数，部分模型可能无法直接运行。


