import os
def load_from_file(path):
    # Read a file
    text = ""
    with open(path, "rt", encoding="utf-8") as in_file:
        text = in_file.read()
    return text