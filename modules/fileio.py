def read_file(path):
    # Read a file
    text = ""
    with open(path, "rt", encoding="utf-8") as in_file:
        text = in_file.read()
    return text

def write_file(path, content):
    # Write content to a file
    with open(path, "wt", encoding="utf-8") as out_file:
        out_file.write(content)