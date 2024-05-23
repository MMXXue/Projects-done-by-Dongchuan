from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
import os
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import ast


def embed_column(column):
    if isinstance(column, str):
        return embeddings.embed_query(column)
    else:
        return column


# 准备种子用户和普通用户的数据
seed_users = pd.read_csv('mgtv曝光点击.csv')  # 种子用户数据
normal_users = pd.read_csv('mgtv曝光未点击.csv')  # 普通用户数据

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')


# 对种子用户的数据进行特征嵌入处理，得到嵌入向量
# seed_users_embeddings = []
# seed_users_embeddings = seed_users.apply(lambda x: x.map(embed_column) if x.dtype == 'object' else x)
# # print(seed_users_embeddings.iloc[0])
# # print(seed_users_embeddings.iloc[1])
# # print(seed_users_embeddings.iloc[2])
# print("seed completed!")
# seed_users_embeddings.to_csv('seed_users_embeddings.csv', index=False)

# normal_users_embeddings = []
# cols_to_drop = ['req_id', 'key', 'request_uri']  # 列名列表，表示哪些列需要被删除
# normal_users = normal_users.drop(cols_to_drop, axis=1)  # 在 DataFrame 中删除指定的列
# normal_users_embeddings = normal_users.apply(lambda x: x.map(embed_column) if x.dtype == 'object' else x)
# print("normal completed")
# normal_users_embeddings.to_csv('normal_users_embeddings.csv', index=False)
#





# 读取种子用户和正常用户的嵌入向量
seed_users_embeddings = pd.read_csv('seed_users_embeddings_one.csv')
normal_users_embeddings = pd.read_csv('normal_users_embeddings.csv')

# 提取特征和目标
X_seed = seed_users_embeddings.drop('item_names', axis=1)
y_seed = seed_users_embeddings['item_names']

# 使用逻辑回归模型训练分类器
model = LogisticRegression()
model.fit(X_seed, y_seed)






# 使用训练好的模型对普通用户的嵌入向量进行预测，得到点击广告的概率
X_normal = pd.DataFrame(normal_users_embeddings, columns=normal_users.columns)
predictions = model.predict_proba(X_normal)[:, 1]  # 获取点击广告的概率

# 筛选出有极大概率点击广告的潜在用户
potential_users = normal_users[predictions > 0.9]  # 设置阈值为0.9
potential_users.to_csv('potential_users.csv', index=False)

# 打印潜在用户
print(potential_users)

