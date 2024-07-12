import unicodedata
import string
import torch
import random
import glob
import random
from collections import namedtuple
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


def read_txt_as_list(file_path: str) -> List:
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
        names = read_txt_as_list(f)
        names_clean = [unicode_to_ascii(name) for name in names]
        country = f.split("/")[-1].split(".txt")[0]
        category_lines[country] = names_clean
        all_categories.append(country)
    return all_categories, category_lines


def letter_to_index(letter: str) -> int:
    """Return the index of letter in ALL_LETTERS"""
    return ALL_LETTERS.find(letter)


def one_hot_encode_letter(letter: str) -> torch.Tensor:
    """One-hot encode a letter."""
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0, letter_to_index(letter)] = 1
    return tensor


def one_hot_encode_text(text: str) -> torch.Tensor:
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
    name_tensor = one_hot_encode_text(name)
    Sample = namedtuple(
        "Sample", ["category", "name", "category_tensor", "name_tensor"]
    )
    return Sample(category, name, category_tensor, name_tensor)


def category_from_output(output: int, all_categories: List) -> str:
    """Return the category with highest value in output."""
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


if __name__ == "__main__":
    print(unicode_to_ascii("Café Münster"))
    print(letter_to_index("a"))
    print(one_hot_encode_letter("a"))
    print(one_hot_encode_text("abc"))
    all_categories, category_lines = load_data()
    print(get_random_training_sample(category_lines, all_categories))
