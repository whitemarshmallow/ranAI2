import pandas as pd
import pymysql
import re
import numpy as np

# 解析 SINR 日志数据的函数
def parse_sinr_logs(filepath):
    pattern = r"\[(.*?)\] INFO: CellID (\d+) - SINR: ([\d.]+) dB"
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                timestamp = pd.to_datetime(match.group(1))
                cell_id = int(match.group(2))
                sinr = float(match.group(3))
                data.append({'timestamp': timestamp, 'cell_id': cell_id, 'SINR': sinr})
    return pd.DataFrame(data)

# 使用示例
sinr_data = parse_sinr_logs('rawdata/SINR_Logs.txt')

# 读取 RSRP 和 RSRQ 数据
rsrp_rsrq_data = pd.read_csv('rawdata/RSRP_RSRQ_Data.csv')

# 转换 RSRP_RSRQ_Data.csv 中的 timestamp 列的格式
rsrp_rsrq_data['timestamp'] = pd.to_datetime(rsrp_rsrq_data['timestamp'], format='%Y/%m/%d %H:%M')

# 合并数据
combined_data = pd.merge(sinr_data, rsrp_rsrq_data, on=['timestamp', 'cell_id'], how='outer')

# 加载 JSON 和 XML 数据
cqi_data = pd.read_json('rawdata/CQIupanddown_data.json')
rsp_latency_data = pd.read_xml('rawdata/RSPLatency.xml')
rsp_latency_data['timestamp'] = pd.to_datetime(rsp_latency_data['timestamp'], format='%Y-%m-%d %H:%M:%S')


print(cqi_data.columns)
print(rsp_latency_data.columns)

# 重命名列名以匹配合并操作中使用的列名
cqi_data.rename(columns={'cellID': 'cell_id'}, inplace=True)
rsp_latency_data.rename(columns={'cellID': 'cell_id'}, inplace=True)

# 确认列名已正确更改
print(cqi_data.columns)
print(rsp_latency_data.columns)

# 再次尝试合并数据
combined_data = pd.merge(combined_data, cqi_data, on=['timestamp', 'cell_id'], how='outer')
combined_data = pd.merge(combined_data, rsp_latency_data, on=['timestamp', 'cell_id'], how='outer')


# 创建数据库连接
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='1234',
    db='db02',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    # 从数据库读取用户数量和流量数据
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM UserCounts")
        user_counts_data = cursor.fetchall()
        user_counts_df = pd.DataFrame(user_counts_data)
        user_counts_df['timestamp'] = pd.to_datetime(user_counts_df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        cursor.execute("SELECT * FROM TrafficData")
        traffic_data = cursor.fetchall()
        traffic_data_df = pd.DataFrame(traffic_data)
        traffic_data_df['timestamp'] = pd.to_datetime(traffic_data_df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    # 合并用户数量数据和流量数据
    combined_data = pd.merge(combined_data, user_counts_df, on=['timestamp', 'cell_id'], how='left')
    combined_data = pd.merge(combined_data, traffic_data_df, on=['timestamp', 'cell_id'], how='left')

    # 填充缺失数据
    combined_data['user_count'] = combined_data['user_count'].fillna(np.random.randint(0, 500))
    combined_data['traffic'] = combined_data['traffic'].fillna(f"{np.random.randint(100, 1000)}GB")

    # 保存最终的合并结果
    combined_data.to_csv('datatransfer/Final_Combined_Data2.csv', index=False)

    print("数据合并完成，输出文件：Final_Combined_Data2.csv")

finally:
    connection.close()  # 关闭数据库连接
