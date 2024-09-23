def get_input_from_list(accepted_input, msg=None):
    if msg is None:
        msg = f'Choose one from {accepted_input}:\n'
    val = None
    while val not in accepted_input:
        val = input(msg)
    return val