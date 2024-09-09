import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np

# 设置基本参数
cell_ids = range(1, 101)  # 基站ID从1到100
start_date = datetime(2023, 9, 1)
hours_duration = 7 * 24  # 7天，每天24小时

# 创建XML根元素
root = ET.Element("network_data")

# 生成时间序列
timestamps = [start_date + timedelta(hours=i) for i in range(hours_duration)]

# 生成数据
for cell_id in cell_ids:
    for timestamp in timestamps:
        data_point = ET.SubElement(root, "data_point")
        ET.SubElement(data_point, "timestamp").text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        ET.SubElement(data_point, "cellID").text = str(cell_id)
        ET.SubElement(data_point, "RSP").text = f"{np.random.uniform(-120, -60):.2f} dBm"  # RSP值范围假设为-120到-60 dBm
        ET.SubElement(data_point, "Latency").text = f"{np.random.uniform(10, 100):.2f} ms"  # Latency假设在10到100毫秒之间

# 生成XML树并保存为XML文件
tree = ET.ElementTree(root)
tree.write('rawdata/RSPLatency.xml')
