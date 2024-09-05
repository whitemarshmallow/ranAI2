import datetime
import numpy as np
import pandas as pd

# 设置随机种子以保证数据的一致性
np.random.seed(42)

# 生成时间序列（每小时，持续7天）
timestamps = pd.date_range(start="2023-09-01", periods=7*24, freq='H')

# 假设有100个基站
cell_ids = np.arange(100)

# 生成SINR日志数据
sinr_log_data = []
log_template = "[{timestamp}] INFO: CellID {cell_id} - SINR: {sinr} dB"

for timestamp in timestamps:
    for cell_id in cell_ids:
        # 生成SINR值
        sinr = np.random.uniform(0, 30)
        # 格式化日志条目
        log_entry = log_template.format(timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"), cell_id=cell_id, sinr=sinr)
        sinr_log_data.append(log_entry)

# 将日志数据保存到文本文件
log_file_path = 'rawdata/SINR_Logs.txt'
with open(log_file_path, 'w') as file:
    file.write('\n'.join(sinr_log_data))

# 输出路径以确认保存位置
print("Log file saved to:", log_file_path)
