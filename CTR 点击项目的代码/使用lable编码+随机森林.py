import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split


data = pd.read_csv('4_一天的数据（点击+未点击频数大于或者等于三的）.csv')


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
clf = RandomForestClassifier(min_samples_split=2, n_estimators=100, max_depth=None)

#####################################################################
# 调优
#####################################################################
# param_grid = {
#     'n_estimators': [10, 25, 50, 100, 200, 300],
#     'max_depth': [3, 5, None],
#     'min_samples_split': [2, 5, 10]
# }
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# print("最佳超参数组合：", grid_search.best_params_)
# print("最佳评估指标值：", grid_search.best_score_)
# 最佳超参数组合： {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
# 最佳评估指标值： 0.92097459101984


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
num_folds = 10

# 创建交叉验证对象
kfold = KFold(n_splits=num_folds)

# 执行交叉验证并计算评估指标
scores = cross_val_score(clf, data, target_total, cv=kfold)

print("交叉验证准确率：", scores)
print("平均准确率：", scores.mean())


#####################################################################
# 数据少的时候：
# The accuracy of the Logistic Regression is (train): 0.9988207547169812
# The accuracy of the Logistic Regression is (test): 0.9107981220657277
# 交叉验证准确率： [0.95305164 0.91037736 0.90566038 0.73113208 0.15566038]
# 平均准确率： 0.7311763663743467
#
#
# 跑一天的：
# The accuracy of the Logistic Regression is (train): 0.9975085643101838
# The accuracy of the Logistic Regression is (test): 0.9539227895392279
# 交叉验证准确率： [0.75124378 0.74875622 0.80099502 0.94278607 0.49127182 0.73815461
#  0.99750623 0.76309227 0.74812968 0.66334165]
# 平均准确率： 0.7645277353878984
#####################################################################
