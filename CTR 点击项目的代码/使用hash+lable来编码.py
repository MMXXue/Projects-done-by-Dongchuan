import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import hashlib

# 从CSV文件中读取数据
data = pd.read_csv('mgtv曝光点击.csv')


def hash_encode(value):
    hashed_value = hashlib.sha256(str(value).encode()).hexdigest()
    return hashed_value

# 读取 CSV 文件
df = pd.read_csv('3_总体.csv')

# 对特定列进行哈希编码
columns_to_encode = ['hour', 'oaid', 'os',  'item_names', 'brand', 'sw', 'sh']  # 选择需要编码的列
for column in columns_to_encode:
    df[column] = df[column].apply(hash_encode)

# # 将编码后的数据保存到新的 CSV 文件
# df.to_csv('hashed_data.csv', index=False)

# 1 点击；0 不点击
ones = np.ones(624)
zeros = np.zeros(437)
target_total = np.concatenate((ones, zeros))  # 将两个数组合并
# print(len(combined_array))

# 读取数据集的 CSV 文件
hash = pd.read_csv('hashed_data.csv')

# 创建 LabelEncoder 对象
le = LabelEncoder()

# 对 DataFrame 中的每个列进行标签编码
for column in hash.columns:
    hash[column] = le.fit_transform(hash[column])

## 保存编码后的数据到新的 CSV 文件
hash.to_csv('encoded_data.csv', index=False)

print(hash)

# encode = pd.read_csv('encoded_data.csv')

# 分割特征和目标变量
# X = hash.drop('item_names', axis=1)
# y = hash['item_names']

X_train, X_test, y_train, y_test = train_test_split(hash, target_total, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=0, solver='liblinear', max_iter=1000)

# 在训练集上训练逻辑回归模型
clf.fit(X_train, y_train)

## 查看其对应的w
print('the weight of Logistic Regression:', clf.coef_)

## 查看其对应的w0
print('the intercept(w0) of Logistic Regression:', clf.intercept_)


## 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

from sklearn import metrics

## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))

## 查看混淆矩阵 (预测值A和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)


# # 获取所有不同特征的特征值
# unique_values = {}
# for column in data.columns:
#     unique_values[column] = data[column].unique()
#
# print(unique_values)
