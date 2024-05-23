import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('4_一天的数据（点击+未点击频数大于或者等于三的）.csv')
normal_users = pd.read_csv('一天的数据的csv文件/similar_users_restored.csv')
potential_users = normal_users.copy()


#####################################################################
# encoded feature and set up target
#####################################################################
le = LabelEncoder()
# 对 DataFrame 中的每个列进行标签编码
for column in data.columns:
    data[column] = le.fit_transform(data[column])
# print(data)

# 1 点击；0 不点击
ones = np.ones(2887)
zeros = np.zeros(1127)
target_total = np.concatenate((ones, zeros))  # 将两个数组合并



#####################################################################
# train and test
#####################################################################
X_train, X_test, y_train, y_test = train_test_split(data, target_total, test_size=0.2, random_state=42)
# 假设你有特征矩阵X和目标变量y
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#####################################################################
# model
#####################################################################
# clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(100,), learning_rate='constant', max_iter=1000, solver='adam')
clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(100, 50, 25), learning_rate='adaptive', max_iter=2000, solver='adam')


#####################################################################
# promote the model
#####################################################################
# param_grid = {
#     'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
#     'activation': ['relu', 'logistic'],
#     'solver': ['adam', 'sgd'],
#     'max_iter':[500, 1000, 2000, 5000],
#     'learning_rate': ['constant', 'adaptive']
# }
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# print("最佳超参数组合：", grid_search.best_params_)
# print("最佳评估指标值：", grid_search.best_score_)
# 最佳超参数组合： {'activation': 'logistic', 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'}
# 最佳评估指标值： 0.8773477201531501

# 最佳超参数组合： {'activation': 'logistic', 'hidden_layer_sizes': (100, 50, 25), 'learning_rate': 'adaptive', 'max_iter': 2000, 'solver': 'adam'}
# 最佳评估指标值： 0.9015852482764302



#####################################################################
# train
#####################################################################
clf.fit(X_train_scaled, y_train)

# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(X_train_scaled)
test_predict = clf.predict(X_test_scaled)

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
# The accuracy of the Logistic Regression is (train): 0.9528301886792453
# The accuracy of the Logistic Regression is (test): 0.9061032863849765
# 交叉验证准确率： [0.86384977 0.86792453 0.8490566  0.49528302 0.14622642]
# 平均准确率： 0.6444680662591905
#
# 一天的数据：
# The accuracy of the Logistic Regression is (train): 0.9009654313298038
# The accuracy of the Logistic Regression is (test): 0.9165628891656289
# 交叉验证准确率： [0.95516812 0.9626401  0.97135741 0.84806974 0.60099751]
# 平均准确率： 0.8676465747213534
#####################################################################



#####################################################################
# use model to choose click user in normal_users
#####################################################################
le = LabelEncoder()
# 对 DataFrame 中的每个列进行标签编码
for column in normal_users.columns:
    normal_users[column] = le.fit_transform(normal_users[column])
# print(normal_users)
normal_users_scaled = scaler.fit_transform(normal_users)

# 预测
predictions = clf.predict(normal_users_scaled)

# 计算概率
proba = clf.predict_proba(normal_users_scaled)
click_proba = proba[:, 1]  # 假设点击的概率对应于第二列
# print(click_proba)

result_df = pd.DataFrame({'click_proba': click_proba})
# print("概率：" + result_df['click_proba'].astype(str))

result_df = pd.concat([potential_users, result_df], axis=1)
# print(result_df)

# 扩散
sorted_indices = np.argsort(-result_df['click_proba'])  # 根据点击概率列降序排序的索引
num_potential_users = int(len(result_df) * 1)  # 想选择的潜在用户数量，这里选择 20%
potential_users_indices = sorted_indices[:num_potential_users]  # 根据排序后的索引选择潜在用户
potential_users = result_df.iloc[potential_users_indices]

potential_users.to_excel('potential_users.xlsx', index=False)





