import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 加载数据
data_path = 'datatransfer/Final_Combined_Data.csv'
data = pd.read_csv(data_path)

# 预处理数据，确保时间格式和类型正确
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 选择一个特定时间点进行分析，这里假设选择数据中的第一个时间点
selected_time = data['timestamp'].iloc[0]
selected_data = data[data['timestamp'] == selected_time]

# 清理和转换数据
selected_data['traffic'] = selected_data['traffic'].str.replace('GB', '').astype(float)  # 转换流量数据为数值
selected_data['user_count'] = pd.to_numeric(selected_data['user_count'], errors='coerce').fillna(0)  # 确保用户数是数值类型

# 设置插值的网格
grid_x, grid_y = np.mgrid[min(selected_data['longitude']):max(selected_data['longitude']):100j,
                          min(selected_data['latitude']):max(selected_data['latitude']):100j]

# 插值
grid_z = griddata((selected_data['longitude'], selected_data['latitude']),
                  selected_data['user_count'], (grid_x, grid_y), method='cubic', fill_value=0)

# 使用 imshow 创建热力图
plt.figure(figsize=(10, 6))
img = plt.imshow(grid_z.T, extent=(min(selected_data['longitude']), max(selected_data['longitude']),
                                    min(selected_data['latitude']), max(selected_data['latitude'])),
                 origin='lower', cmap='hot')
cb = plt.colorbar(img, label='Interpolated User Count')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Interpolated User Count Heatmap at {selected_time}')
plt.grid(True)

# 保存图像
plt.savefig('datatransfer/Interpolated_User_Count_Heatmap.png', format='png', dpi=300)
plt.show()
