import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('data.csv')[['petal length', 'petal width']]
labels_true = pd.read_csv('data.csv')['class']

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

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
# 定义参数范围
eps_range = np.arange(0.1, 1.0, 0.03)
minPts_range = range(3, 50)

# 存储最佳参数和分数
best_ari = -1
best_params = {'eps': None, 'minPts': None}

# 循环遍历参数组合
for eps in eps_range:
    for minPts in minPts_range:
        clusters = dbscan_manual(data_scaled, eps, minPts)
        if len(set(clusters)) > 1:  # 确保有多于一个聚类（避免全部为噪声）
            ari = adjusted_rand_score(labels_true, clusters)
            if ari > best_ari:
                best_ari = ari
                best_params['eps'] = eps
                best_params['minPts'] = minPts

# 输出最佳参数
print("最佳参数组合:", best_params)
print("最佳 ARI 分数:", best_ari)
