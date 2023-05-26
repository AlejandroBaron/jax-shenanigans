import random
import string


def random_string(length):
    available_chars = string.ascii_letters + string.digits + "_-"
    return "".join(random.choices(available_chars, k=length))
