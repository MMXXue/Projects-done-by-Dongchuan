import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances_argmin_min, pairwise_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('一天的数据的csv文件/正样本.csv')
original = pd.read_csv('一天的数据的csv文件/feature_b.csv')

le = LabelEncoder()
# 对 DataFrame 中的每个列进行标签编码
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# 提取前500个特征，并进行标准化处理
features_a = data.iloc[:500].values
features_b = data.iloc[500:].values
# 标准化处理
scaler = StandardScaler()
features_a = scaler.fit_transform(features_a)
features_b = scaler.transform(features_b)

# 计算相似度矩阵
similarity_matrix = pairwise_distances(features_b, features_a)
print(similarity_matrix)
print(similarity_matrix.shape)

# 找到每行的最大相似值及其索引
max_similarities = np.max(similarity_matrix, axis=1)
max_indices = np.argmax(similarity_matrix, axis=1)

# 创建一个 DataFrame，存储最大相似值、索引和对应的 feature_b 用户
df = pd.DataFrame({'Max Similarity': max_similarities, 'Index of Max Similarity': max_indices})
print(df)

# 取前500名用户的数据
top_2000_users = df.nlargest(500, 'Max Similarity')
# print(top_2000_users)
print(top_2000_users.index.values)

selected_indices = top_2000_users.index.values
selected_feature_b = original.iloc[selected_indices]
print(selected_feature_b)

# # 将数据保存为 Excel 文件
df_selected_feature_b = pd.DataFrame(selected_feature_b)
df_selected_feature_b.to_excel('top_2000_feature_b.xlsx', index=False)
