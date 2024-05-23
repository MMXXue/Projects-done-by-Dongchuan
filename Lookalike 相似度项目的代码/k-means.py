import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('一天的数据的csv文件/正样本.csv')
original = pd.read_csv('一天的数据的csv文件/普通用户_没有新增列版本.csv')
original_n = original.copy()

# 去除重复
data = data.drop_duplicates()
original = original.drop_duplicates()
print("去除重复finish---------")

# 标签编码
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

for column in original.columns:
    original[column] = le.fit_transform(original[column])
print("编码标签finish---------")

# 取值
features_a = data.iloc[:].values
features_b = original.iloc[:].values
print("取值finish---------")

# 标准化处理
scaler = StandardScaler()
features_a = scaler.fit_transform(features_a)
features_b = scaler.fit_transform(features_b)
print("标准化finish---------")

# 使用K-means算法进行聚类
kmeans = KMeans(n_clusters=len(features_a))
kmeans.fit(features_b)
print("聚类finish---------")

# 获取每个簇的中心点
cluster_centers = kmeans.cluster_centers_

# 选择与数据集a中用户最相似的用户
similar_users = []
num_similar_users = 30
index = []

for user_a in features_a:
    distances = np.linalg.norm(cluster_centers - user_a, axis=1)
    similar_cluster_index = np.argmin(distances)
    # 在该簇中找到与用户a最相似的多个用户
    users_in_cluster = np.where(kmeans.labels_ == similar_cluster_index)[0]
    distances_within_cluster = np.linalg.norm(features_b[users_in_cluster] - user_a, axis=1)
    similar_user_indices_within_cluster = users_in_cluster[np.argsort(distances_within_cluster)[:num_similar_users]]
    print(similar_user_indices_within_cluster)
    index.append(similar_user_indices_within_cluster)
    similar_users_within_cluster = features_b[similar_user_indices_within_cluster]
    print(similar_users_within_cluster)
    similar_users.append(similar_users_within_cluster)
    print('_____________')

print("找用户finish---------")

index = np.concatenate(index)
sorted_index = np.sort(index)
sorted_index = np.unique(sorted_index)

# 创建一个空的DataFrame来存放结果
result_df = pd.DataFrame(columns=original_n.columns)

# 循环逐行取出数据并存放在新的DataFrame中
for i in sorted_index:
    row_data = original_n.iloc[i]
    result_df = pd.concat([result_df, pd.DataFrame(row_data).T], ignore_index=True)

# 打印结果DataFrame
print(result_df)

# 将结果DataFrame保存为Excel文件
result_df.to_excel("similar_users_restored.xlsx", index=False)

