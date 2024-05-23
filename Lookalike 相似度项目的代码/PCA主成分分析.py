import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('一天的数据的csv文件/普通用户_没有新增列版本.csv')

# 假设你的数据存储在名为data的DataFrame中，其中包含要进行主成分分析的特征列
features = ['hour', 'os', 'item_names', 'brand', 'sw', 'sh', 'key']

le = LabelEncoder()
# 对 DataFrame 中的每个列进行标签编码
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# 创建StandardScaler对象并对数据进行标准化
data_to_scale = data[features]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_scale)


# 创建PCA对象，指定要保留的主成分数量
n_components = 5  # 假设你想保留两个主成分
pca = PCA(n_components=n_components)
# 使用PCA计算主成分
principal_components = pca.fit_transform(scaled_data)

# 获取解释方差比例
explained_variance_ratio = pca.explained_variance_ratio_

# 计算累积方差比例
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 绘制方差比例的累积图
plt.rcParams['font.family'] = 'Arial Unicode MS'  # 替换为你希望使用的字体名称
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('主成分数量')
plt.ylabel('累积方差比例')
plt.title('方差比例的累积图')

# 添加拐点线
desired_components = 4  # 设置拐点对应的主成分数量
plt.axvline(x=n_components, color='r', linestyle='--', label='拐点')

# 显示图形
plt.legend()
plt.show()