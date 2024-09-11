import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据导入
red_wine_path = 'processed_red_wine.csv'  # 更新为你的红葡萄酒数据集的路径
white_wine_path = 'processed_white_wine.csv'  # 更新为你的白葡萄酒数据集的路径

red_wine_df = pd.read_csv(red_wine_path)
white_wine_df = pd.read_csv(white_wine_path)

# 红葡萄酒的箱线图绘制
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(data=red_wine_df, x='quality', y='alcohol', ax=axes[0], palette='Reds').set_title('Alcohol vs Quality (Red Wine)')
sns.boxplot(data=red_wine_df, x='quality', y='volatile acidity', ax=axes[1], palette='Reds').set_title('Volatile Acidity vs Quality (Red Wine)')
sns.boxplot(data=red_wine_df, x='quality', y='total sulfur dioxide', ax=axes[2], palette='Reds').set_title('Total Sulfur Dioxide vs Quality (Red Wine)')

plt.tight_layout()
plt.show()  # 确保能够显示图像

# 白葡萄酒的箱线图绘制
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(data=white_wine_df, x='quality', y='alcohol', ax=axes[0], palette='Blues').set_title('Alcohol vs Quality (White Wine)')
sns.boxplot(data=white_wine_df, x='quality', y='density', ax=axes[1], palette='Blues').set_title('Density vs Quality (White Wine)')
sns.boxplot(data=white_wine_df, x='quality', y='volatile acidity', ax=axes[2], palette='Blues').set_title('Volatile Acidity vs Quality (White Wine)')

plt.tight_layout()
plt.show()  # 确保能够显示图像
