import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score




data = pd.read_csv('4_一天的数据（点击+未点击频数大于或者等于三的）.csv')
# data = pd.read_csv('3_总体_减少了特征.csv')


#####################################################################
# encoded feature and set up target
#####################################################################
le = LabelEncoder()
# 对 DataFrame 中的每个列进行标签编码
for column in data.columns:
    data[column] = le.fit_transform(data[column])
print(data)

# 1 点击；0 不点击
ones = np.ones(2887)
zeros = np.zeros(1127)
target_total = np.concatenate((ones, zeros))  # 将两个数组合并


#####################################################################
# train and test
#####################################################################
X_train, X_test, y_train, y_test = train_test_split(data, target_total, test_size=0.2, random_state=42)


#####################################################################
# model
#####################################################################
clf = KNeighborsClassifier(metric='manhattan', n_neighbors=7, weights='distance')

#####################################################################
# 调优
#####################################################################
# param_grid = {
#     'n_neighbors': [3, 5, 7],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# print("最佳超参数组合：", grid_search.best_params_)
# print("最佳评估指标值：", grid_search.best_score_)
# # 最佳超参数组合： {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}
# # 最佳评估指标值： 0.8931764557685693
#
#
#
# #####################################################################
# # train
# #####################################################################
clf.fit(X_train, y_train)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is (train):', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is (test):', metrics.accuracy_score(y_test, test_predict))
#
# #####################################################################
# # K折
# #####################################################################
# 定义交叉验证的折数（一般取5或10）
num_folds = 5

# 创建交叉验证对象
kfold = KFold(n_splits=num_folds)

# 执行交叉验证并计算评估指标
scores = cross_val_score(clf, data, target_total, cv=kfold)

print("交叉验证准确率：", scores)
print("平均准确率：", scores.mean())


#####################################################################
# The accuracy of the Logistic Regression is (train): 0.8714622641509434
# The accuracy of the Logistic Regression is (test): 0.892018779342723
# 交叉验证准确率： [0.96244131 0.90566038 0.90566038 0.42924528 0.15566038]
# 平均准确率： 0.6717335459296659

# 一天的数据：
# The accuracy of the Logistic Regression is (train): 0.9975085643101838
# The accuracy of the Logistic Regression is (test): 0.8891656288916563
# 交叉验证准确率： [0.71980075 0.72976339 0.67372354 0.80323786 0.26932668]
# 平均准确率： 0.6391704425114052
#####################################################################
