import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif

# 指定红葡萄酒和白葡萄酒数据集的路径
red_wine_path = 'winequality-red.csv'
white_wine_path = 'winequality-white.csv'

# 使用pd.read_csv函数读取数据集
red_wine = pd.read_csv(red_wine_path, delimiter=';')
white_wine = pd.read_csv(white_wine_path, delimiter=';')

# 移除红葡萄酒数据集中的重复记录
red_wine.drop_duplicates(inplace=True)

# 移除白葡萄酒数据集中的重复记录
white_wine.drop_duplicates(inplace=True)

# 计算总酸度并添加为新列
red_wine['total_acidity'] = red_wine['fixed acidity'] + red_wine['volatile acidity']
white_wine['total_acidity'] = white_wine['fixed acidity'] + white_wine['volatile acidity']

# 数据转换
## 规范化“质量”数据到[0,1]范围内
scaler = MinMaxScaler()
red_wine['quality_normalized'] = scaler.fit_transform(red_wine[['quality']])
white_wine['quality_normalized'] = scaler.fit_transform(white_wine[['quality']])

# 使用分位数确定分割点
red_quantiles = red_wine['fixed acidity'].quantile([0.25, 0.75]).tolist()
white_quantiles = white_wine['fixed acidity'].quantile([0.25, 0.75]).tolist()

# 离散化“固定酸度”
red_wine['fixed_acidity_levels'] = pd.cut(red_wine['fixed acidity'], bins=[0] + red_quantiles + [float('inf')], labels=['low', 'medium', 'high'], right=False)
white_wine['fixed_acidity_levels'] = pd.cut(white_wine['fixed acidity'], bins=[0] + white_quantiles + [float('inf')], labels=['low', 'medium', 'high'], right=False)

# 准备特征和目标变量
X_red = red_wine.drop(['quality', 'quality_normalized', 'fixed_acidity_levels', 'total_acidity'], axis=1)
y_red = red_wine['quality']
X_white = white_wine.drop(['quality', 'quality_normalized', 'fixed_acidity_levels', 'total_acidity'], axis=1)
y_white = white_wine['quality']

# 执行ANOVA F-test
f_values_red, p_values_red = f_classif(X_red, y_red)
f_values_white, p_values_white = f_classif(X_white, y_white)

# 将F值和p值与特征名关联
red_features_pvalues = pd.DataFrame({'Feature': X_red.columns, 'F-value': f_values_red, 'p-value': p_values_red})
white_features_pvalues = pd.DataFrame({'Feature': X_white.columns, 'F-value': f_values_white, 'p-value': p_values_white})

# 根据F值选择影响最大的前三个特征
top_red_features = red_features_pvalues.nlargest(3, 'F-value')['Feature'].tolist()
top_white_features = white_features_pvalues.nlargest(3, 'F-value')['Feature'].tolist()

# 打印结果
print("对红葡萄酒质量评级影响最显著的前三个特征：", top_red_features)
print("对白葡萄酒质量评级影响最显著的前三个特征：", top_white_features)

# 保存处理后的数据集
red_wine.to_csv('processed_red_wine.csv', index=False)
white_wine.to_csv('processed_white_wine.csv', index=False)


