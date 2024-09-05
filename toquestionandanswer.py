import pandas as pd

# 加载数据
data_path = 'datatransfer/Final_Combined_Data.csv'
data = pd.read_csv(data_path)

# 预处理数据
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['traffic'] = data['traffic'].str.replace('GB', '').astype(float)

# 生成问答对
qa_pairs = []
for _, row in data.iterrows():
    question = f"On {row['timestamp'].date()}, how was the traffic and user count at cell ID {row['cell_id']}?"
    answer = f"Traffic was {row['traffic']} GB, user count was {row['user_count']}, and fault status was {'present' if row['fault'] == 'TRUE' else 'not present'}."
    qa_pairs.append((question, answer))

# 保存问答对到TXT文件
qa_path = 'datatransfer/qa_pairs.txt'
with open(qa_path, 'w') as file:
    for q, a in qa_pairs:
        file.write(f"Q: {q}\nA: {a}\n\n")

