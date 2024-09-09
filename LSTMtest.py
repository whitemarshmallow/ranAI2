import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 加载数据集
data = pd.read_csv('datatransfer/Final_Combined_Data2.csv')
print(data.head())  # 打印数据的前五行以检查数据的加载情况

data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values('timestamp', inplace=True)
data.reset_index(drop=True, inplace=True)

# 假设流量数据列名为 'traffic'
# data['traffic'] = pd.to_numeric(data['traffic'], errors='coerce').fillna(method='ffill')
data['traffic'] = data['traffic'].str.replace('GB', '').astype(float)

# 标准化
scaler = MinMaxScaler(feature_range=(0, 1))
data['scaled_traffic'] = scaler.fit_transform(data[['traffic']].values)

# 生成序列数据
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5  # 使用5个时间步长作为输入
X, y = create_sequences(data['scaled_traffic'].values, seq_length)

# 划分训练测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为PyTorch张量
X_train = torch.tensor(X_train).float().unsqueeze(-1)
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float().unsqueeze(-1)
y_test = torch.tensor(y_test).float()

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # 选择最后一个时间步的输出用于预测
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions



model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    single_loss = loss_function(y_pred, y_train)
    single_loss.backward()
    optimizer.step()
    if epoch%5 == 1:
        print(f'Epoch {epoch} loss: {single_loss.item()}')


model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = loss_function(preds, y_test)
    print(f'Test Loss: {test_loss.item()}')

