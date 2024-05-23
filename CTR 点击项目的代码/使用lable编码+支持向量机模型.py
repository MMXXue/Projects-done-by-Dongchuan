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
clf = SVC()

#####################################################################
# 调优
#####################################################################
# param_grid = {
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': ['scale', 'auto']
# }
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# print("最佳超参数组合：", grid_search.best_params_)
# print("最佳评估指标值：", grid_search.best_score_)


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
# The accuracy of the Logistic Regression is (train): 0.9033018867924528
# The accuracy of the Logistic Regression is (test): 0.8685446009389671
# 交叉验证准确率： [0.83098592 0.75943396 0.77358491 0.19811321 0.14622642]
# 平均准确率： 0.5416688812117991

# 一天的数据：
# The accuracy of the Logistic Regression is (train): 0.8364995328558081
# The accuracy of the Logistic Regression is (test): 0.8393524283935243
# 交叉验证准确率： [0.87048568 0.83188045 0.84557908 0.73474471 0.        ]
# 平均准确率： 0.65653798256538
#####################################################################
