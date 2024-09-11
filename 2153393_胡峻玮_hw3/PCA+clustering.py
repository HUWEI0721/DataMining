import heapq
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from heapq import heappush, heappop

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 加载数据集
data_path = 'data.csv'
data = pd.read_csv(data_path)

# 提取数值特征，忽略类别标签
X = data.iloc[:, :-1].values

# 初始化簇：开始时，每个数据点是一个独立的簇
def initialize_clusters(points):
    return [[point] for point in points]

# 计算簇间的距离
def compute_distances(clusters):
    distances = np.zeros((len(clusters), len(clusters)))
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):  # 只计算矩阵的一半，因为它是对称的
            if i != j:
                # 使用欧几里得距离公式计算两个簇的距离
                distances[i][j] = np.linalg.norm(np.mean(clusters[i], axis=0) - np.mean(clusters[j], axis=0))
                distances[j][i] = distances[i][j]  # 填充矩阵的另一半
    return distances

# 找到最近的两个簇
def find_closest_clusters(distances):
    min_distance = np.inf
    closest_clusters = None
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):  # 避免重复和对角线
            if distances[i][j] < min_distance:
                min_distance = distances[i][j]
                closest_clusters = (i, j)
    return closest_clusters

# 合并最近的两个簇
def merge_clusters(clusters, closest_clusters):
    i, j = closest_clusters
    merged_cluster = clusters[i] + clusters[j]
    new_clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in closest_clusters]
    new_clusters.append(merged_cluster)
    return new_clusters


def improved_hierarchical_clustering(X, n_clusters=3):
    N = len(X)
    # 计算初始的距离矩阵
    distances = squareform(pdist(X, 'euclidean'))
    # 优先队列，存储(距离, 簇索引i, 簇索引j)
    queue = [(distances[i][j], i, j) for i in range(N) for j in range(i + 1, N)]
    heapq.heapify(queue)

    # 初始化簇
    clusters = {i: [i] for i in range(N)}

    while len(clusters) > n_clusters:
        # 找到最近的两个簇
        dist, i, j = heappop(queue)
        # 合并这两个簇
        if i in clusters and j in clusters:  # 确保这两个簇仍然有效
            new_cluster = clusters[i] + clusters[j]
            new_index = max(clusters.keys()) + 1  # 创建新簇索引
            clusters[new_index] = new_cluster
            del clusters[i], clusters[j]

            # 更新距离矩阵，只针对新簇
            for k in clusters.keys():
                if k != new_index:
                    new_dist = np.mean([distances[x][y] for x in clusters[new_index] for y in clusters[k]])
                    heappush(queue, (new_dist, new_index, k))

    return list(clusters.values())

# 执行层次聚类
clusters = improved_hierarchical_clustering(X, 3)
# 使用PCA降维以便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 准备一个颜色数组，为不同的簇分配不同的颜色
colors = ['r', 'g', 'b']

# 绘制每个簇的点
plt.figure()
for cluster_index, cluster in enumerate(clusters):
    # 假设cluster包含的是数据点的索引，我们从X中获取这些数据点
    cluster_points = np.array([X[i] for i in cluster])
    # 此时，cluster_points是一个二维数组，每一行代表一个数据点
    # 因此，我们可以直接对它应用PCA变换
    cluster_pca = pca.transform(cluster_points)
    plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], color=colors[cluster_index], label=f'Cluster {cluster_index+1}')

plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('层次聚类结果可视化')
plt.show()

