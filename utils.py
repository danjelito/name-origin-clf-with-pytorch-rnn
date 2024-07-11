import unicodedata
import string
import torch
import random
import glob
import random
from typing import List, Dict

ALL_LETTERS = string.ascii_letters + " .,;"
N_LETTERS = len(ALL_LETTERS)


def unicode_to_ascii(text: str) -> str:
    """Removes special alphabet character and return ASCII version."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def read_file_as_list(file_path: str) -> List:
    """Read lines in txt file, return as list."""
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


def load_data() -> Dict:
    """Returns {country1: [names], country2: [names]...}"""
    path = "input/names/"
    txt_files = glob.glob(path + "*.txt")
    category_lines = {}
    all_categories = []
    for f in txt_files:
        names = read_file_as_list(f)
        names_clean = [unicode_to_ascii(name) for name in names]
        country = f.split("/")[-1].split(".txt")[0]
        category_lines[country] = names_clean
        all_categories.append(country)
    return all_categories, category_lines


def letter_to_index(letter: str) -> int:
    """Return the index of letter in ALL_LETTERS"""
    return ALL_LETTERS.find(letter)


def letter_to_tensor(letter: str) -> torch.Tensor:
    """One-hot encode a letter."""
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0, letter_to_index(letter)] = 1
    return tensor


def text_to_tensor(text: str) -> torch.Tensor:
    """One-hot encode a text"""
    tensor = torch.zeros(len(text), 1, N_LETTERS)
    for i, letter in enumerate(text):
        tensor[i, 0, letter_to_index(letter)] = 1
    return tensor


def get_random_training_sample(category_lines, all_categories):
    """Return a random (category, name)."""
    # select random category
    category = random.choice(all_categories)
    # select random name from this category
    name = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.int64)
    name_tensor = text_to_tensor(name)
    return category, name, category_tensor, name_tensor


if __name__ == "__main__":
    print(unicode_to_ascii("Café Münster"))
    print(letter_to_index("a"))
    print(letter_to_tensor("a"))
    print(text_to_tensor("abc"))
    all_categories, category_lines = load_data()
    print(get_random_training_sample(category_lines, all_categories))
