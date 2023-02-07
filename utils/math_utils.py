import string
import random


def is_power_of_two(n: int):
    return n != 0 and ((n & (n - 1)) == 0)


def random_string(string_length: int = 10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(string_length))
