import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score, KFold


data = pd.read_csv('3_总体_未删除特征.csv')
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
ones = np.ones(624)
zeros = np.zeros(437)
target_total = np.concatenate((ones, zeros))  # 将两个数组合并


#####################################################################
# train and test
#####################################################################
X_train, X_test, y_train, y_test = train_test_split(data, target_total, test_size=0.2, random_state=42)


#####################################################################
# model
#####################################################################
# clf = LogisticRegression(random_state=0, solver='liblinear', max_iter=1000)
clf = LogisticRegression(max_iter=1000, C = 0.1, solver = 'lbfgs')


#####################################################################
# 超参数调优：网格
#####################################################################
# 定义超参数的候选值
# param_grid = {
#     'C': [0.1, 1.0, 10.0],
#     'solver': ['liblinear', 'lbfgs', 'sag', 'saga']
# }
# # 创建网格搜索对象
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# # 执行网格搜索
# grid_search.fit(data, target_total)
# # 输出最佳超参数组合和对应的评估指标值
# print("最佳超参数组合：", grid_search.best_params_)
# print("最佳评估指标值：", grid_search.best_score_)

# 最佳超参数组合： {'C': 0.1, 'solver': 'lbfgs'}
# 最佳评估指标值： 0.8313136681725574


#####################################################################
# train model
#####################################################################
# 在训练集上训练逻辑回归模型
clf.fit(X_train, y_train)

## 查看其对应的w
print('the weight of Logistic Regression:', clf.coef_)
## 查看其对应的w0
print('the intercept(w0) of Logistic Regression:', clf.intercept_)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is (train):', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is (test):', metrics.accuracy_score(y_test, test_predict))

## 查看混淆矩阵 (预测值A和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

#####################################################################
# K折
#####################################################################
# 定义交叉验证的折数（一般取5或10）
num_folds = 5

# 创建交叉验证对象
kfold = KFold(n_splits=num_folds)

# 执行交叉验证并计算评估指标
scores = cross_val_score(clf, data, target_total, cv=kfold)

print("交叉验证准确率：", scores)
print("平均准确率：", scores.mean())

#####################################################################
# The accuracy of the Logistic Regression is (train): 0.8455188679245284
# The accuracy of the Logistic Regression is (test): 0.8732394366197183
# 交叉验证准确率： [0.82159624 0.83490566 0.81132075 0.5754717  0.26415094]
# 平均准确率： 0.6614890601470458
#####################################################################
