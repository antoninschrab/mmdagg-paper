def generate_seed(*args):
    """
    Generate an integer by concatenating arguments, different arguments
    always give a different integer. 
    We assume that all arguments are digits except the last one which 
    can be an integer consisting of at most 4 digits.
    """
    str_args = [str(x) for x in args]
    if len(str_args[-1]) == 1:
        str_args[-1] = "".join(["000", str_args[-1]])
    if len(str_args[-1]) == 2:
        str_args[-1] = "".join(["00", str_args[-1]])
    if len(str_args[-1]) == 3:
        str_args[-1] = "".join(["0", str_args[-1]])
    return int("".join(str_args))
