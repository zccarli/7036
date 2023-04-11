
import torch
import torch.nn as nn


class RNN(nn.Module):
    #     self train embedding
    #     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
    #         super().__init__()
    #         self.embedding = nn.Embedding(input_dim, embedding_dim)
    #         self.rnn = nn.RNN(embedding_dim, hidden_dim)
    #         self.fc = nn.Linear(hidden_dim, output_dim)

    #   pretrained embedding
    def __init__(self, wordVectors, embedding_dim, hidden_dim, num_layers, output_dim, maxseqlength):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(wordVectors).float()) # word2vec
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers)  # similar to one hidden layer
        self.fc = nn.Linear(maxseqlength * hidden_dim, output_dim)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.maxseqlength = maxseqlength
        self.ln = nn.LayerNorm([maxseqlength, embedding_dim], eps=1e-05, elementwise_affine=True) # layer normalization

        def _weights_init(m):
            """ kaiming init (https://arxiv.org/abs/1502.01852v1)"""
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)


    def forward(self, x): # text = [batch size, sent len]

        embedded = self.embedding(x) # embedded = [batch size, sent len, emb dim]
        embedded = self.ln(embedded)
        out, hidden = self.rnn(embedded) # output = [batch size, sent len, hid dim] # hidden = [num_layers, sent len, hid dim]
        out = out.view(out.size(0), self.maxseqlength * self.hidden_dim)
        # out = out[:, -1, :].squeeze(1)
        out = self.fc(out) # output = [batch size, output size]
        # output = torch.sigmoid(output)  # torch.nn.Sigmoid, torch.nn.functional.sigmoid()
        # output = nn.functional.softmax(output, dim=1)
        return out



class LSTM(nn.Module):
    def __init__(self, wordVectors, input_dim, hidden_dim, num_layers, output_dim, maxseqlength):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(wordVectors).float())
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.maxseqlength = maxseqlength
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # () is the weight size   *2 due to birectional

        def _weights_init(m):
            """ kaiming init (https://arxiv.org/abs/1502.01852v1)"""
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        embedded = self.embedding(x) # embedded = [batch size, sent len, emb dim]
        out, (hn, cn) = self.lstm(embedded, (h0.detach(), c0.detach())) # output = [batch size, sent len, hid dim * 2] # hidden = [num_layers * 2, batch, hid dim]
        # out = out.view(out.size(0), self.maxseqlength * self.hidden_dim * 2)
        out = out[:, -1, :].squeeze(1)
        out = self.fc(out)  # output = [batch size, output size]
        return out


