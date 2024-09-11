from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


# 加载数据集
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# 执行PCA降维，为了可视化方便
pca = PCA(n_components=2)
iris_pca_score = pca.fit_transform(X_iris)

# 执行层次聚类
Z_iris = linkage(X_iris, method='ward', metric='euclidean')

# 确定聚类数，这里我们已知是3
n_clusters = 3
iris_labels = fcluster(Z_iris, t=n_clusters, criterion='maxclust')

# 可视化
plt.figure(figsize=(8, 6))
for cluster, marker in zip(range(1, n_clusters+1), ['x', 'o', '+']):
    x_axis = iris_pca_score[:, 0][iris_labels == cluster]
    y_axis = iris_pca_score[:, 1][iris_labels == cluster]
    plt.scatter(x_axis, y_axis, marker=marker)

plt.title('Hierarchical Clustering of Iris Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# 模型评估
silhouette_avg = silhouette_score(X_iris, iris_labels)

plt.show(), silhouette_avg
