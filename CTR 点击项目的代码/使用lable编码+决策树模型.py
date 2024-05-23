import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


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
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2)

#####################################################################
# 调优
#####################################################################
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'min_samples_split': [2, 5, 10],
#     'criterion': ['gini', 'entropy']
# }
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# print("最佳超参数组合：", grid_search.best_params_)
# print("最佳评估指标值：", grid_search.best_score_)
# 最佳超参数组合： {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2}
# 最佳评估指标值： 0.9080055690915418

#####################################################################
# train
#####################################################################
clf.fit(X_train, y_train)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is (train):', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is (test):', metrics.accuracy_score(y_test, test_predict))

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
# The accuracy of the Logistic Regression is (train): 0.9257075471698113
# The accuracy of the Logistic Regression is (test): 0.8873239436619719
# 交叉验证准确率： [0.98122066 0.95283019 0.93867925 0.28301887 0.15566038]
# 平均准确率： 0.6622818673044557
#####################################################################