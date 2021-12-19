import torch
import numpy as np


def tensor_id_size(input_tensor):
    return input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)


# axis=0 : frequency, axis=1 : time
def alteration_tensor_1d(input_tensor, axis=0):
    batch_size, row, column = tensor_id_size(input_tensor)
    for index in range(batch_size):
        if axis == 0:
            random_index = random_uniform_integer(0, row)
            input_tensor[index, random_index] = 0
        elif axis == 1:
            random_index = random_uniform_integer(0, column)
            input_tensor[index, :, random_index] = 0
    return input_tensor


def random_alteration_tensor_1d(input_tensor):
    batch_size, row, column = tensor_id_size(input_tensor)
    for index in range(batch_size):
        axis = random_uniform_integer(0, 3)
        if axis == 0:
            random_index = random_uniform_integer(0, row)
            input_tensor[index, random_index] = 0
        elif axis == 1:
            random_index = random_uniform_integer(0, column)
            input_tensor[index, :, random_index] = 0
        elif axis == 2:
            random_index = random_uniform_integer(0, row)
            input_tensor[index, random_index] = 0
            random_index = random_uniform_integer(0, column)
            input_tensor[index, :, random_index] = 0
    return input_tensor


def random_uniform_integer(start, end):
    return int(np.random.uniform(start, end, 1)[0])


if __name__ == '__main__':
    input_data = torch.randn(8, 8, 5)
    output_data = random_alteration_tensor_1d(input_data)
    print(output_data)