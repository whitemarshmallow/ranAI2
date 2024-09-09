import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import geopy.distance
import torch

# 节点添加：为每个基站添加一个节点，并存储相关属性。
# 边添加：根据基站之间的地理距离添加边，条件是距离小于 5 公里。
# 可视化：使用 matplotlib 和 networkx 进行网络图的绘制，节点颜色代表故障状态（红色表示有故障，绿色表示无故障），大小反映了用户数量。

# 加载数据
data_path = 'datatransfer/Final_Combined_Data.csv'
data = pd.read_csv(data_path)

# 创建图
G = nx.Graph()

# 添加节点和属性
for idx, row in data.iterrows():
    G.add_node(row['cell_id'],
               timestamp=row['timestamp'],
               latitude=row['latitude'],
               longitude=row['longitude'],
               RSRP=row['RSRP'],
               RSRQ=row['RSRQ'],
               fault=row['fault'],
               model=row['model'],
               user_count=row['user_count'],
               traffic=row['traffic'])

# 添加边，此处可以根据实际情况添加逻辑，比如相邻的基站或特定条件下的基站关系
for node1 in G.nodes(data=True):
    for node2 in G.nodes(data=True):
        if node1 != node2:
            dist = geopy.distance.geodesic((node1[1]['latitude'], node1[1]['longitude']),
                                           (node2[1]['latitude'], node2[1]['longitude'])).km
            if dist < 5:  # 假设距离小于5公里的基站有边
                G.add_edge(node1[0], node2[0], distance=dist)


# 假设 G 是你的图对象
pos = nx.spring_layout(G)  # 为节点设置布局

# 节点和边的可视化属性
node_sizes = [G.nodes[node]['user_count']*10 for node in G.nodes]  # 调整节点大小
node_colors = ['red' if G.nodes[node]['fault'] else 'green' for node in G.nodes]  # 节点颜色
labels = {node: f"Station {node}" for node in G.nodes}

# 边的标签
edge_labels = nx.get_edge_attributes(G, 'distance')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)  # 调整边标签字体大小



import torch
from geometric.data import Data
from geometric.utils import from_networkx

# 将 NetworkX 图转化为 PyTorch Geometric 数据结构
# 节点特征: RSRP, RSRQ, user_count, traffic
for node in G.nodes:
    G.nodes[node]['feature'] = [
        G.nodes[node]['RSRP'],
        G.nodes[node]['RSRQ'],
        G.nodes[node]['user_count'],
        G.nodes[node]['traffic']
    ]

# 标签: fault (0 无故障, 1 有故障)
for node in G.nodes:
    G.nodes[node]['label'] = int(G.nodes[node]['fault'])

# 将 NetworkX 图转化为 PyTorch Geometric 格式
data = from_networkx(G)

# 将特征和标签转换为 PyTorch 张量
data.x = torch.tensor([G.nodes[node]['feature'] for node in G.nodes], dtype=torch.float)
data.y = torch.tensor([G.nodes[node]['label'] for node in G.nodes], dtype=torch.long)

