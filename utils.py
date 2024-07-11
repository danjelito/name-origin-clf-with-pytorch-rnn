import unicodedata
import string
import torch
import random
import glob

ALL_LETTERS = string.ascii_letters + " .,;"
N_LETTERS = len(ALL_LETTERS)


def unicode_to_ascii(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def read_file_as_list(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


def load_data():
    path = "input/names/"
    txt_files = glob.glob(path + "*.txt")
    category_lines = {}
    for f in txt_files:
        names = read_file_as_list(f)
        names_clean = [unicode_to_ascii(name) for name in names]
        country = f.split("/")[-1].split(".txt")[0]
        category_lines[country] = names_clean
    return category_lines


if __name__ == "__main__":
    name = "Café Münster"
    print(unicode_to_ascii(name))
    print(load_data())
