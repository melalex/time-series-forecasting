from torch import nn


class StockLstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StockLstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        return self.fc(lstm_out)
