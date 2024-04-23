import numpy as np
import torch

def convolution(input, conv_filter, biases, padding = 0, stride = 1):
    channels, height, width = input.shape
    padded_image = np.pad(input, pad_width=[(0, 0), (padding, padding), (padding, padding)], mode="constant")
    n_filters, filter_channels, filter_height, filter_width = conv_filter.shape

    assert filter_channels == channels

    output_shape = (
        n_filters,
        (height + 2 * padding - filter_height) // stride + 1,
        (width + 2 * padding - filter_width) // stride + 1,
    )
    output = np.zeros(output_shape)

    for m in range(output_shape[0]):
        for x in range(output_shape[1]):
            for y in range(output_shape[2]):
                output[m][x][y] = biases[m]
                for i in range(filter_height):
                    for j in range(filter_width):
                        for k in range(filter_channels):
                            output[m][x][y] += padded_image[k][x + i][y + j] * conv_filter[m][k][i][j]
    
    return output

def convolution_test(input, padding = 0, stride = 1):
    input = np.moveaxis(input, -1, 0)
    channels, height, width = input.shape
    n_filters = [1, 2, 3]
    kernel_sizes = [1, 3, 5, 7]

    for n_filter in n_filters:
        for kernel_size in kernel_sizes:
            conv_filter = np.random.random_integers(low=0, high=10, size=(n_filter, channels, kernel_size, kernel_size))
            biases = np.random.random_integers(low=0, high=10, size=(n_filter))
            # manually written convolution
            result_to_test = convolution(input, conv_filter=conv_filter, biases=biases, padding=padding, stride=stride)
            
            # convolution using torch for getting reference
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            input_torch = torch.from_numpy(input)
            conv_filter_torch = torch.from_numpy(conv_filter)
            biases_torch = torch.from_numpy(biases)
            
            input_torch = input_torch.to(torch.float32)
            conv_filter_torch = conv_filter_torch.to(torch.float32)
            biases_torch = biases_torch.to(torch.float32)

            input_torch.to(device)
            conv_filter_torch.to(device)
            biases_torch.to(device)

            result_to_compare = torch.nn.functional.conv2d(input_torch,
                                                           conv_filter_torch,
                                                           biases_torch,
                                                           stride=stride,
                                                           padding=padding)
            result_to_compare = result_to_compare.detach().cpu().numpy()
            result_to_compare = result_to_compare.astype(np.uint8)
            result_to_test = result_to_test.astype(np.uint8)

            # if assertion passed then 2 arrays are equal
            assert (result_to_compare == result_to_test).all()
            print(f'Passed convolution test:\n\tParams: n_filter = {n_filter}, kernel_size = ({kernel_size}, {kernel_size})')
