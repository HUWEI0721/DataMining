DBSCAN 聚类算法实现


这个代码实现了一个手动实现的 DBSCAN（Density-Based Spatial Clustering of Applications with Noise）聚类算法，并提供了一些评估聚类结果的指标。DBSCAN 是一种密度聚类算法，可以识别具有足够高密度的区域，并将其视为聚类，同时能够将低密度区域视为噪声。

使用方法
1. 准备数据：确保已经准备好要聚类的数据集，并且该数据集包含 'petal length' 和 'petal width' 两个特征。
2. 运行代码：运行 dbscan_manual 函数来执行 DBSCAN 算法。你可以在 eps 和 minPts 参数中调整聚类的敏感度和最小点数。
3. 可视化结果：执行代码后，将会显示一个散点图，显示了聚类的结果。
4. 评估聚类：代码还提供了三个指标来评估聚类结果：纯度（purity）、标准化互信息（NMI）和调整后的兰德指数（ARI）。

文件结构
DBSCAN.py: 包含了 DBSCAN 算法的实现代码。
data.csv: 包含了用于聚类的数据集。
README.md: 说明文档，提供了项目的概述、使用方法等信息。

依赖
Python 3.11
numpy
pandas
matplotlib
scikit-learn

 执行示例
python DBSCAN.py


作者：胡峻玮
邮箱：2153393@tongji.edu.cn

