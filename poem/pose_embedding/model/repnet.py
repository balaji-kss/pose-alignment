import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x 


class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, max_len=64):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                    nhead = n_head,
                                                    dim_feedforward = dim_ff,
                                                    dropout = dropout,
                                                    activation = 'relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
                
    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op

class RepNetHead_not_as_paper(torch.nn.Module):
    def __init__(self, max_seq_len, device='cpu'):
        super().__init__()

        self.device = device
        self.max_seq_len = max_seq_len

        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.max_seq_len * 32, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.transEncoder1 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers=1, max_len=max_seq_len)
        self.transEncoder2 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers=1, max_len=max_seq_len)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.ln1_2 = nn.LayerNorm(512)
        self.fc1_2 = nn.Linear(512, self.max_seq_len//2)
        self.fc1_3 = nn.Linear(self.max_seq_len//2, 1)


        #periodicity prediction
        self.fc2_1 = nn.Linear(512, 512)
        self.ln2_2 = nn.LayerNorm(512)
        self.fc2_2 = nn.Linear(512, self.max_seq_len//2)
        self.fc2_3 = nn.Linear(self.max_seq_len//2, 1)

        self.to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn2(self.conv3x3(x)))   #batch, 32, num_frame, num_frame
        x = self.dropout1(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.max_seq_len, -1)  #batch, num_frame, 32*num_frame
        x = self.ln1(F.relu(self.input_projection(x)))  #batch, num_frame, d_model=512
        x = x.transpose(0, 1)                           #num_frame, batch, d_model=512

        #period
        x1 = self.transEncoder1(x)
        y1 = x1.transpose(0, 1)
        y1 = F.relu(self.ln1_2(self.fc1_1(y1)))
        y1 = F.relu(self.fc1_2(y1))
        y1 = F.relu(self.fc1_3(y1))

        #periodicity
        x2 = self.transEncoder2(x)
        y2 = x2.transpose(0, 1)
        y2 = F.relu(self.ln2_2(self.fc2_1(y2)))
        y2 = F.relu(self.fc2_2(y2))
        y2 = F.relu(self.fc2_3(y2)) 

        return y1, y2


class RepNetHead(torch.nn.Module):
    def __init__(self, max_seq_len, device='cpu'):
        super().__init__()

        self.device = device
        self.max_seq_len = max_seq_len

        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.max_seq_len * 32, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512*4, num_layers=3, max_len=max_seq_len)
        self.fc1 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        
        #period length prediction
        self.fc1_2 = nn.Linear(512, self.max_seq_len//2)
        self.fc1_3 = nn.Linear(self.max_seq_len//2, 1)


        #periodicity prediction
        self.fc2_2 = nn.Linear(512, self.max_seq_len//2)
        self.fc2_3 = nn.Linear(self.max_seq_len//2, 1)

        self.to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn2(self.conv3x3(x)))   #batch, 32, num_frame, num_frame
        x = self.dropout1(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.max_seq_len, -1)  #batch, num_frame, 32*num_frame
        x = self.ln1(F.relu(self.input_projection(x)))  #batch, num_frame, d_model=512
        x = x.transpose(0, 1)                           #num_frame, batch, d_model=512

        x = self.transEncoder(x)
        y = x.transpose(0, 1)
        y = F.relu(self.ln2(self.fc1(y)))

        #period
        y1 = F.relu(self.fc1_2(y))
        y1 = F.relu(self.fc1_3(y1))

        #periodicity
        y2 = F.relu(self.fc2_2(y))
        y2 = F.relu(self.fc2_3(y2)) 

        return y1, y2


class MultiRepHead(nn.Module):
    def __init__(self, max_seq_len_list):
        super(MultiRepHead, self).__init__()
        self.heads = nn.ModuleList([RepNetHead(max_seq_len) for max_seq_len in max_seq_len_list])

    def forward(self, x):
        # x : list of different scale self-similarity-matrices
        pred_period, pred_periodicity= [], []
        for sim, head in zip(x, self.heads):
            y1, y2 = head(sim)
            pred_period.append(y1)
            pred_periodicity.append(y2)            
        return pred_period, pred_periodicity

class SelfSimilarity(torch.nn.Module):
    def __init__(self, temperature=13.544, device='cuda'):
        super(SelfSimilarity, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, x, temperature=13.544):
        # Compute the squared norms of each input vector
        x_squared_norms = (x ** 2).sum(dim=-1, keepdim=True)

        # Compute the squared Euclidean distances using the formula:
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x * y
        out = x_squared_norms + x_squared_norms.transpose(1, 2) - 2 * x.matmul(x.transpose(1, 2))

        out = out.unsqueeze(1)  # insert a new dimension [n,s,s] --> [n,1,s,s]

        # Apply the softmax function
        out = F.softmax(-out / temperature, dim=-1)

        return out
