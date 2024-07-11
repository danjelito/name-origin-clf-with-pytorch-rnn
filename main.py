import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import ALL_LETTERS, N_LETTERS
from utils import load_data


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(
            in_features=input_size + hidden_size, out_features=hidden_size
        )
        self.i2o = nn.Linear(
            in_features=input_size + hidden_size, out_features=output_size
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_t, hidden_t):
        combined = torch.cat((input_t, hidden_t), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# load data
all_categories, category_lines = load_data()
n_categories = len(all_categories)

# setup rnn
rnn = RNN(input_size=N_LETTERS, hidden_size=128, output_size=n_categories)
