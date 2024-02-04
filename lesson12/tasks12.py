import torch
import time


def im2col(picture, kernel_width, kernel_height):
    (channels_num, image_width, image_height) = picture.shape
    new_w = image_width - kernel_width + 1
    new_h = image_height - kernel_height + 1
    columns = torch.zeros((new_w * new_h, channels_num * kernel_width * kernel_height))
    for column in range(new_w):
        for line in range(new_h):
            columns[column * new_h + line] = picture[..., column:column + kernel_width, line:line + kernel_height].ravel()
    return torch.Tensor(columns).T


def fast_conv2d(picture, kernel):
    (kernel_channels, kernel_width, kernel_height) = kernel.shape
    (_, picture_width, picture_height) = picture.shape
    picture_reshaped = im2col(picture, kernel_width, kernel_height)
    kernel = kernel.view((1, kernel_channels * kernel_width * kernel_height))
    result = kernel @ picture_reshaped
    result = result.view(1, picture_width - kernel_width + 1, picture_height - kernel_height + 1)
    return result


def main():
    picture = torch.IntTensor([[list(range(1, 5)),
                                list(range(5, 9)),
                                list(range(9, 13)),
                                list(range(13, 17))],
                               [list(range(17, 21)),
                                list(range(21, 25)),
                                list(range(25, 29)),
                                list(range(29, 33))],
                               [list(range(33, 37)),
                                list(range(37, 41)),
                                list(range(41, 45)),
                                list(range(45, 49))],
                               ])
    print("Picture: \n", picture)
    picture_reshaped = im2col(picture, 2, 2)
    print("Picture_reshaped: \n", picture_reshaped)

    picture = torch.Tensor(1, 3, 80, 60).random_(0, 255)
    kernel = torch.Tensor([[[-0.1, -0.1, -0.1],
                            [-0.1,    2, -0.1],
                            [-0.1, -0.1, -0.1]],
                           [[-0.1, 0.1, -0.1],
                            [0.1, 0.5, 0.1],
                            [-0.1, 0.1, -0.1]],
                           [[-0.1, 0.2, -0.1],
                            [0.2, 1, 0.2],
                            [-0.1, 0.2, -0.1]]])
    start = time.time()
    result = fast_conv2d(picture[0, ...], kernel)
    finish = time.time()
    print("The result of fast_conv2d: ", finish - start, " seconds!")
    print("Shape: \n", result.shape)
    torch_conv = torch.nn.Conv2d(3, 1, 3)
    start = time.time()
    result = torch_conv(picture)
    finish = time.time()
    print("The result of torch_conv2d: ", finish - start, " seconds!")
    print("Shape: \n", result.shape)


main()
