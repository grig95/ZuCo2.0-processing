import re

def get_input_from_list(accepted_input, msg=None):
    if msg is None:
        msg = f'Choose one from {accepted_input}:\n'
    val = None
    while val not in accepted_input:
        val = input(msg)
    return val


def normalize_word(word): # removes punctuation from the beginning and end and lowers the capitalization
    return re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', word).lower()