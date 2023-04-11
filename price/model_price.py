
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        # self.bn1 = nn.BatchNorm1d(8)
        # self.bn2 = nn.BatchNorm1d(16)
        # self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        out = self.conv_1(x)   # x [batch size, 1,  7]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv_2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv_3(out)
        out = self.relu(out)
        out = self.dropout(out) # out [batch size, kernel number, 7]
        out = out.view(out.size(0), -1) # out [batch size, kernel number * 7]
        out = nn.Linear(out.size(1), 1)(out)  # out [batch size, 1]

        return out


class RNN(nn.Module):
    #     self train embedding
    #     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
    #         super().__init__()
    #         self.embedding = nn.Embedding(input_dim, embedding_dim)
    #         self.rnn = nn.RNN(embedding_dim, hidden_dim)
    #         self.fc = nn.Linear(hidden_dim, output_dim)

    #   pretrained embedding
    def __init__(self, embedding_dim, hidden_dim, num_layers, output_dim, history):
        super().__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers)  # similar to one hidden layer
        self.fc = nn.Linear(history * hidden_dim, output_dim)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.history = history
        self.ln = nn.LayerNorm([history, embedding_dim], eps=1e-05, elementwise_affine=True) # layer normalization

        def _weights_init(m):
            """ kaiming init (https://arxiv.org/abs/1502.01852v1)"""
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)


    def forward(self, x): # text = [batch size, sent len]
         # embedded = [batch size, sent len, emb dim]
        # embedded = self.ln(x)
        out, hidden = self.rnn(x) # output = [batch size, sent len, hid dim] # hidden = [1, sent len, hid dim]
        out = out.view(out.size(0), self.history * self.hidden_dim)
        # out = out[:, -1, :].squeeze(1)
        out = self.fc(out) # output = [batch size, output size]
        # output = torch.sigmoid(output)  # torch.nn.Sigmoid, torch.nn.functional.sigmoid()
        # output = nn.functional.softmax(output, dim=1)
        return out
# class CNN2d
# class LSTM
