import numpy as np
import torch

def im2col_conv(image, conv_filter, biases, padding = 0, stride = 1):
    channels, height, width = image.shape
    padded_image = np.pad(image, pad_width=[(0, 0), (padding, padding), (padding, padding)], mode="constant")
    n_filters, filter_channels, filter_height, filter_width = conv_filter.shape

    assert filter_channels == channels

    output_shape = (
        n_filters,
        (height + 2 * padding - filter_height) // stride + 1,
        (width + 2 * padding - filter_width) // stride + 1,
    )

    output = np.zeros(output_shape)

    for fy in range(filter_height):
        y_max = fy + stride * output_shape[1]
        for fx in range(filter_width):
            x_max = fx + stride * output_shape[2]
            convolved_region = padded_image[:, fy:y_max:stride, fx:x_max:stride]
            for filter_id in range(n_filters):
                output[filter_id] += np.sum(convolved_region * conv_filter[filter_id, :, fy, fx][:, np.newaxis, np.newaxis], axis=0)
    
    for filter_id in range(n_filters):
        output[filter_id] += biases[filter_id]

    return output

def im2col_conv_test(input, padding = 0, stride = 1):
    input = np.moveaxis(input, -1, 0)
    channels, height, width = input.shape
    n_filters = [1, 2, 3]
    kernel_sizes = [1, 3, 5, 7]

    for n_filter in n_filters:
        for kernel_size in kernel_sizes:
            conv_filter = np.random.random_integers(low=0, high=10, size=(n_filter, channels, kernel_size, kernel_size))
            biases = np.random.random_integers(low=0, high=10, size=(n_filter))
            # getting result of im2col convolution
            result_to_test = im2col_conv(input, conv_filter=conv_filter, biases=biases, padding=padding, stride=stride)
            
            # getting reference result
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
            print(f'Passed im2col convolution test:\n\tParameters: n_filter = {n_filter}, kernel_size = ({kernel_size}, {kernel_size})')
