import pandas as pd
import numpy as np

# 设定随机种子
np.random.seed(42)

# 生成时间序列（每小时，持续7天）
timestamps = pd.date_range(start="2023-09-01", periods=7*24, freq='H')

# 生成100个基站的信息
cell_ids = np.arange(100)
latitudes = np.random.uniform(-90, 90, size=100)
longitudes = np.random.uniform(-180, 180, size=100)
models = ["Model_A", "Model_B", "Model_C"]

# 扩展时间序列和基站ID，以创建数据框架
data = []
for timestamp in timestamps:
    for cell_id in cell_ids:
        # 随机选择基站型号
        model = np.random.choice(models)
        # 生成RSRP和RSRQ值
        rsrp = np.random.uniform(-120, -60)
        rsrq = np.random.uniform(-20, -3)
        # 随机标记是否出现故障
        fault = np.random.choice([True, False], p=[0.1, 0.9])
        # 获取对应基站的经纬度
        latitude = latitudes[cell_id]
        longitude = longitudes[cell_id]
        # 组合所有信息
        data.append([timestamp, cell_id, latitude, longitude, rsrp, rsrq, fault, model])

# 转换为DataFrame
columns = ["timestamp", "cell_id", "latitude", "longitude", "RSRP", "RSRQ", "fault", "model"]
df = pd.DataFrame(data, columns=columns)

# 显示数据框架的头部

csv_file_path = "rawdata/RSRP_RSRQ_Data.csv"
df.to_csv(csv_file_path, index=False)
