import torch
import torch.nn as nn

def keep_variance(x, min_variance):
    return x + min_variance

class ThreeLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(ThreeLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
  
        return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    
# class ThreeLayerLSTM_FC(nn.Module):
#     def __init__(self, input_size_lstm, input_size_fc, hidden_size, num_layers, output_size, dropout):
#         super(ThreeLayerLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(input_size_lstm, hidden_size, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.fc1 = nn.Linear(input_size_fc, 10)
#         self.fc2 = nn.Linear(hidden_size + 10, hidden_size)  # 输入维度是 LSTM 隐藏层维度 + 全连接层输出维度
#         self.fc3 = nn.Linear(hidden_size, output_size)

#     def forward(self, x_lstm, x_fc):
#         h0 = torch.zeros(self.num_layers, x_lstm.size(0), self.hidden_size).to(x_lstm.device)
#         c0 = torch.zeros(self.num_layers, x_lstm.size(0), self.hidden_size).to(x_lstm.device)

#         out_lstm, _ = self.lstm(x_lstm, (h0, c0))
#         out_fc = self.fc1(x_fc)

#         # 在同一维度上进行 concatenate
#         out_concat = torch.cat((out_lstm[:, -1, :], out_fc), dim=1)

#         out = self.fc2(out_concat)
#         out = self.fc3(out)
#         return out
