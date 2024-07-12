import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import N_LETTERS
from utils import (
    load_data,
    get_random_training_sample,
    category_from_output,
)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN, self).__init__()
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


# define training step
def train(name_tensor, category_tensor):
    hidden_tensor = rnn.init_hidden()
    # iterate through each char in name
    for i in range(name_tensor.shape[0]):
        output, hidden_tensor = rnn(name_tensor[i], hidden_tensor)
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


# load data
all_categories, category_lines = load_data()
n_categories = len(all_categories)

# setup rnn
rnn = RNN(input_size=N_LETTERS, hidden_size=64, output_size=n_categories)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(
    rnn.parameters(),
    lr=0.005,
)

# training loop
current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iter = 200_000
for i in range(n_iter):
    # get 1 random name
    sample = get_random_training_sample(category_lines, all_categories)
    category = sample.category
    name = sample.name
    category_tensor = sample.category_tensor
    name_tensor = sample.name_tensor
    # train
    output, loss = train(name_tensor, category_tensor)
    current_loss += loss
    # append loss
    if (i + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
    # print loss
    if (i + 1) % print_steps == 0:
        guess = category_from_output(output, all_categories)
        correct = "CORRECT" if guess == category else f"WRONG"
        print(
            f"Iteration: {i + 1}/{n_iter} ({(i + 1) / n_iter:.0%}) | Loss: {loss:.4f} | Name: {name} | Predicted: {guess} | Actual: {category} | Result: {correct}"
        )


plt.figure()
plt.plot(all_losses)
plt.show()

# # example: one step
# input_tensor = text_to_tensor("Albert")
# hidden_tensor = rnn.init_hidden()
# output , next_hidden_tensor = rnn(input_tensor[0], hidden_tensor)
# print(output.shape)
# print(next_hidden_tensor.shape)
