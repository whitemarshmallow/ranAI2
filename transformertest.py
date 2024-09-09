import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math


# Define a simple Transformer model
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.ninp = ninp
        from torch.nn import Transformer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.transformer = Transformer(d_model=ninp, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers, dim_feedforward=nhid, dropout=dropout)
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer(src, src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)




import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def add_word(self, word):
        if word not in self.stoi:
            self.stoi[word] = len(self.itos)
            self.itos[self.stoi[word]] = word

    def __len__(self):
        return len(self.itos)

class QADataset(Dataset):
    def __init__(self, qa_pairs, vocab):
        self.qa_pairs = qa_pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        question = [self.vocab.stoi.get(word, self.vocab.stoi["<UNK>"]) for word in question.split()]
        answer = [self.vocab.stoi.get(word, self.vocab.stoi["<UNK>"]) for word in answer.split()]
        return torch.tensor([self.vocab.stoi["<SOS>"]] + question + [self.vocab.stoi["<EOS>"]]), torch.tensor([self.vocab.stoi["<SOS>"]] + answer + [self.vocab.stoi["<EOS>"]])

def load_data(file_path):
    qa_pairs = []
    vocab = Vocabulary()
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Q:'):
                question = line.strip().split('Q: ')[1]
            elif line.startswith('A:'):
                answer = line.strip().split('A: ')[1]
                qa_pairs.append((question, answer))
                words = question.split() + answer.split()
                for word in words:
                    vocab.add_word(word)
    return qa_pairs, vocab

qa_pairs, vocab = load_data('datatransfer/qa_pairs.txt')
dataset = QADataset(qa_pairs, vocab)
loader = DataLoader(dataset, batch_size=2, shuffle=True)




model = TransformerModel(ntoken=len(vocab), ninp=256, nhead=8, nhid=512, nlayers=3, dropout=0.5)
model.train()
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(1):
    total_loss = 0
    for question, answer in loader:
        optimizer.zero_grad()
        output = model(question.T, answer.T[:-1])
        loss = criterion(output.reshape(-1, len(vocab)), answer.T[1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(loader)}')



def generate_answer(model, question, vocab, max_len=50):
    model.eval()
    inputs = torch.tensor([vocab.stoi["<SOS>"]] + [vocab.stoi.get(word, vocab.stoi["<UNK>"]) for word in question.split()] + [vocab.stoi["<EOS>"]], dtype=torch.long).unsqueeze(0)
    device = next(model.parameters()).device  # 获取模型使用的设备
    inputs = inputs.to(device)

    outputs = [vocab.stoi["<SOS>"]]
    for i in range(max_len):
        current_input = torch.tensor(outputs, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(current_input, current_input)  # 假设模型已经可以处理同样长度的输入和输出
            next_token = logits[:, -1, :].argmax(-1).item()  # 取最后一个输出的最大值作为下一个 token
        outputs.append(next_token)
        if next_token == vocab.stoi["<EOS>"]:
            break

    translated_sentence = [vocab.itos[token] for token in outputs if token not in (vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"])]
    return ' '.join(translated_sentence)

# 示例用法:
question = "On 2023-09-01, how was the traffic and user count at cell ID 43?"
print(generate_answer(model, question, vocab))


# def generate_answer(model, question, vocab, max_len=50):
#     model.eval()
#     question = torch.tensor([vocab.stoi["<SOS>"]] + [vocab.stoi.get(word, vocab.stoi["<UNK>"]) for word in question.split()] + [vocab.stoi["<EOS>"]])
#     question = question.unsqueeze(1)
#     with torch.no_grad():
#         answer = model.generate(question, max_length=max_len)[0]
#         answer = ' '.join([vocab.itos[token] for token in answer if token not in (vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"])])
#     return answer
#
# # Example usage:
# question = "On 2023-09-01, how was the traffic and user count at cell ID 43?"
# print(generate_answer(model, question, vocab))
