import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 加载数据
data_path = 'datatransfer/Final_Combined_Data.csv'
data = pd.read_csv(data_path)

# 预处理数据
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['traffic'] = data['traffic'].str.replace('GB', '').astype(float)
data['fault'] = data['fault'].replace({'FALSE': 0, 'TRUE': 1})  # 将故障状态转换为数值

# 选择特征和目标变量
features = data[['traffic', 'user_count', 'latitude', 'longitude']]
target = data['fault']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 创建LightGBM数据结构
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 训练模型
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# 预测
y_pred = bst.predict(X_test)
y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]

# 评估模型
accuracy = accuracy_score(y_test, y_pred_binary)
conf_matrix = confusion_matrix(y_test, y_pred_binary)
class_report = classification_report(y_test, y_pred_binary)

# 打印评估结果
print("accuracy+"+str(accuracy))
print("conf_matrix+"+str(conf_matrix))
print("class_report+"+str(class_report))
