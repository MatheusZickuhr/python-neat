import math


def sigmoid(value):
    return 1 / (1 + math.e ** -value)


def get_activation_from_str(string):
    if string == 'sigmoid':
        return sigmoid
    else:
        raise Exception(f' \'{string}\' activation function not found')
