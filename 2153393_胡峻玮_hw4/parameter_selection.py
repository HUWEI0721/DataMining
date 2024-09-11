import time

# 记录程序开始时间
start_time = time.time()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree

# 假设数据已正确加载且仅包含 'petal length' 和 'petal width'
data = pd.read_csv('data.csv')[['petal length', 'petal width']]
labels_true = pd.read_csv('data.csv')['class']
# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# DBSCAN 算法实现
def dbscan_manual(D, eps, minPts):
    labels = np.full(shape=len(D), fill_value=-1)  # 所有点初始化为 -1 (未分类)
    cluster_id = 0
    visited = np.zeros(len(D), dtype=bool)
    tree = KDTree(D)

    def region_query(p_index):
        return tree.query_radius([D[p_index]], eps)[0]

    def expand_cluster(p_index, neighbors):
        labels[p_index] = cluster_id
        i = 0
        while i < len(neighbors):
            n_index = neighbors[i]
            if not visited[n_index]:
                visited[n_index] = True
                n_neighbors = region_query(n_index)
                if len(n_neighbors) >= minPts:
                    neighbors = np.append(neighbors, n_neighbors)
            if labels[n_index] == -1:
                labels[n_index] = cluster_id
            i += 1

    for i in range(len(D)):
        if not visited[i]:
            visited[i] = True
            P_neighbors = region_query(i)
            if len(P_neighbors) >= minPts:
                expand_cluster(i, P_neighbors)
                cluster_id += 1

    return labels

# 参数选择
eps = 0.4  # 根据数据和肘部图选择适当的值
minPts = 35  # 可调整以改善聚类效果

# 执行 DBSCAN
clusters = dbscan_manual(data_scaled, eps, minPts)

# 可视化结果
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('Manual DBSCAN Clustering')
plt.xlabel('Scaled Petal Length')
plt.ylabel('Scaled Petal Width')
plt.colorbar(label='Cluster Label')
plt.show()

def purity_score(y_true, y_pred):
    # 创建一个数据框架来比较真实标签和预测的聚类
    contingency_matrix = pd.crosstab(y_true, y_pred)
    # 对于每个聚类，选择最多的真实标签的数量，然后求和并除以总数
    purity = np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix.sum())
    return purity

# 计算纯度
purity = purity_score(labels_true, clusters)
print("纯度:", purity)
from sklearn.metrics import normalized_mutual_info_score

# 计算 NMI
nmi = normalized_mutual_info_score(labels_true, clusters)
print("标准化互信息 (NMI):", nmi)
from sklearn.metrics import adjusted_rand_score

# 计算 ARI
ari = adjusted_rand_score(labels_true, clusters)
print("调整后的兰德指数 (ARI):", ari)


# 记录程序结束时间
end_time = time.time()

# 计算程序运行时间
run_time = end_time - start_time

print("程序运行时间为：", run_time, "秒")