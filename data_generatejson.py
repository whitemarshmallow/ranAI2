import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置基本参数
cell_ids = range(1, 101)  # 基站ID从1到100
start_date = datetime(2023, 9, 1)
hours_duration = 7 * 24  # 7天，每天24小时

# 生成时间序列
timestamps = [start_date + timedelta(hours=i) for i in range(hours_duration)]

# 初始化数据列表
data_list = []

# 生成数据
for cell_id in cell_ids:
    for timestamp in timestamps:
        data_point = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "cellID": cell_id,
            "CQI": np.random.randint(1, 16),  # CQI值范围假设为1到15
            "Downlink_Throughput": np.random.uniform(10, 100),  # 下行速率假设在10到100 Mbps之间
            "Uplink_Throughput": np.random.uniform(1, 50)  # 上行速率假设在1到50 Mbps之间
        }
        data_list.append(data_point)

# 生成JSON文件
with open('rawdata/CQIupanddown_data.json', 'w') as file:
    json.dump(data_list, file, indent=4)
