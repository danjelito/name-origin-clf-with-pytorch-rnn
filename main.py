import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import ALL_LETTERS, N_LETTERS
from utils import (
    load_data,
    letter_to_tensor,
    text_to_tensor,
    get_random_training_sample,
)


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


def category_from_output(output):
    """Return the category with highest value in output."""
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


# define training step
def train(text_tensor, category_tensor):
    hidden_tensor = rnn.init_hidden()
    # iterate through each char in name
    for i in range(text_tensor.shape[0]):
        output, hidden_tensor = rnn(text_tensor[i], hidden_tensor)
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
    lr=0.001,
)

# training loop
current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iter = 200_000
for i in range(n_iter):
    category, name, category_tensor, name_tensor = get_random_training_sample(
        category_lines, all_categories
    )
    output, loss = train(name_tensor, category_tensor)
    current_loss += loss
    if (i + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
    if (i + 1) % print_steps == 0:
        guess = category_from_output(output)
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
