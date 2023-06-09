import math
import numpy as np
import cv2
from matplotlib import pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 316552496


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    # We will use the function convolve from numpy, that get 3 parameters:
    # a - First 1D input array,
    # v - Second 1D input array,
    # Mode - 3 options:
    #   {‘full’ - output shape: (a+v-1 , 1),
    #   ‘valid’ - output shape: max(a, v) - min(a, v) + 1,
    #   ‘same’ - output shape: Max(a,v) }
    return np.convolve(in_signal, k_size, "full")


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    # We will use the function filter2D from cv2, that get the parameters:
    # src – input image.
    # ddepth – desired depth of the dst image, -1 means the same as src.depth()
    # kernel - the kernel, a single-channel floating point matrix
    # dst – output image of the same size and the same number of channels as src.
    # anchor – indicates the relative position of a filtered point within the kernel, (-1, -1) is the kernel center
    # delta – optional value added to the filtered pixels before storing them in dst.
    # borderType – pixel extrapolation method

    return cv2.filter2D(src=in_image, ddepth=-1, kernel=kernel,
                        dst=None, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    # we could write only:     convolved_image = cv2.filter2D(in_image, -1, kernel)
    # Note that cv2.filter2d (d instead of D) does Correlation and not Convolution!!!


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """
    derivative = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])

    # Compute row derivative using convolution with [1, 0, -1] kernel
    dx = conv2D(in_image, derivative)

    # Compute column derivative using convolution with [1, 0, -1]T kernel
    dy = conv2D(in_image, derivative.T)

    # Compute magnitude and direction
    magnitude = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    direction = np.arctan2(dy, dx)

    return direction, magnitude


def calculateGaussian1cell(x, y, sigma):
    """                                                 Y
    Gaussian mask will be created like this:     -1    0    1
                                             -1 [-1,-1|-1,0|-1,1]
                                           X  0 [0,-1 |0,0 |0,1 ]
                                              1 [1,-1 |1,0 |1,1 ]
        the center will be the origin (0,0) and the other cells is relatively to it
    :param x: the x parameter of the cell, for example -1
    :param y: the y parameter of the cell, for example 1
    :param sigma: Aka Std, if sigma becomes smaller - the Gaussian filter will become more peaky
            which means more weight to the central pixels
    :return: float number [0,1]:
    """
    return (1 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))


def GaussianFilter(window_size, sigma):
    maxX = -(window_size // 2)  # We want to avoid reaching to the edges of the image
    minX = -maxX
    maxY = maxX
    minY = minX

    G = np.zeros((window_size, window_size))
    for x in range(minX, maxX + 1):
        for y in range(minY, maxY + 1):
            v = calculateGaussian1cell(x, y, sigma)
            G[x - minX, y - minY] = v
    return G


def kernel_sigma(kernel_size: int) -> float:
    return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel, Using the previous functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    X, Y = in_image.shape
    for row in range(X):
        for col in range(Y):
            in_image[row][col] = calculateGaussian1cell(row, col, 1)
    return in_image


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel1D = cv2.getGaussianKernel(k_size, kernel_sigma(k_size))  # 1D array
    kernel2D = np.dot(kernel1D, kernel1D.T)  # 2D array
    # print(np.sum(G2))  # =>The sum of all the values in this 2D matrix is 1 or very close

    return conv2D(in_image, kernel2D)

    # We could use also:     return cv2.filter2D(in_image, -1, kernel2D, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    dev_simple_matrix = np.array([[-1, 0, 1], [0, 0, 0], [-1, 0, 1]])
    after_lap_conv = conv2D(img, dev_simple_matrix)
    return zeroCrossSearcher(after_lap_conv)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # Define the Laplacian of Gaussian (LoG) filter
    # The Laplacian is 2nd order derivative
    laplacian_matrix = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Smooth with 2D Gaussian
    newImage = blurImage2(img, 5)

    # Apply Laplacian filter - Convolve the image with the LoG filter
    newImage = conv2D(newImage, laplacian_matrix)

    # Find zero crossings
    return zeroCrossSearcher(newImage)


def zeroCrossSearcher(mat: np.ndarray) -> np.ndarray:
    edges = np.zeros_like(mat)

    for i in range(1, mat.shape[0] - 1):
        for j in range(1, mat.shape[1] - 1):
            if mat[i, j] == 0:
                edges[i, j] = 255
            elif (mat[i - 1, j] * mat[i + 1, j] < 0) or (mat[i, j - 1] * mat[i, j + 1] < 0):
                edges[i, j] = 255

    return edges


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles, [(x,y,radius),(x,y,radius),...]
    """
    img = img.astype(np.uint8)  # Convert image to np.uint8

    edges = cv2.Canny(img, 50, 150)  # Detect edges using Canny edge detection

    rows, cols = img.shape
    dMax = int((rows ** 2 + cols ** 2) ** 0.5)
    accumulator = np.zeros((rows, cols, dMax), dtype=np.uint64)  # Use uint64 for accumulator

    for ri in range(rows):
        for ci in range(cols):
            if edges[ri, ci] > 0:  # Only process edge pixels
                for r in range(min_radius, max_radius + 1):
                    for theta in range(360):
                        a = ri - int(r * np.cos(np.deg2rad(theta)))  # Calculate center x-coordinate
                        b = ci - int(r * np.sin(np.deg2rad(theta)))  # Calculate center y-coordinate
                        if 0 <= a < rows and 0 <= b < cols:
                            accumulator[a, b, r] += 1  # Increment accumulator for each possible circle

    circles = []
    for a in range(rows):
        for b in range(cols):
            for r in range(min_radius, max_radius + 1):
                if accumulator[a, b, r] > 0:  # If accumulator value is above threshold, consider it a circle
                    circles.append((a, b, r))

    return circles

# def f_houghCircles(img):
#     numRows, numCols = img.shape[0], img.shape[1]
#     dMax = int((numRows ** 2 + numCols ** 2) ** 0.5)
#     newImage = np.zeros((numRows, numCols, dMax))
#     idx = np.argwhere(img)
#     r, c = idx[:, 0], idx[:, 1]
#     for i in range(len(r)):
#         for a in range(numRows):
#             for b in range(numCols):
#                 ri, ci = r[i], c[i]
#                 di = int(((ri - a) ** 2 + (ci - b) ** 2) ** 0.5)
#                 if 0 < di < dMax:
#                     newImage[a, b, di] += 1
#     return newImage

    # rows, cols = img.shape
    # dMax = int((rows**2 + cols**2)**0.5)
    # H = np.zeros((rows, cols, dMax))
    # idx = np.argwhere(img)
    # r,c = idx[:,0], idx[:,1]
    # for i in range(len(r)):
    #     for a in range(rows):
    #         for b in range(cols):
    #             ri, ci = r[i], c[i]
    #             di = int(((ri-a)**2 + (ci-b)**2)**0.5)
    #             if di > 0 and di <dMax:
    #                 H[a,b,di]+=1
    # return H


def bilateral(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float):
    img = cv2.imread('eye.jpg', cv2.IMREAD_GRAYSCALE) / 255.0
    y, x = 300, 667
    pivot_v = img[y, x]
    neighborhood = img[
                   y - k_size:y + k_size + 1,
                   x - k_size:x + k_size + 1]
    sigma = .01
    diff = pivot_v - neighborhood
    diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma))
    gaus = cv2.getGaussianKernel(2 * k_size + 1, k_size)
    gaus = gaus.dot(gaus.T)
    combo = gaus * diff_gau
    result = combo * neighborhood / combo.sum()


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    in_image = in_image.astype('float32')

    # OpenCV implementation
    opencv_filtered_image = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)

    # Custom implementation
    height, width = in_image.shape
    my_filtered_image = np.zeros_like(in_image, dtype=np.float32)

    # Pad the input image
    padded_image = np.pad(in_image, k_size // 2, mode='constant')

    # Calculate spatial Gaussian kernel
    spatial_kernel = np.zeros((k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            d = np.sqrt(((i - k_size // 2) ** 2 + (j - k_size // 2) ** 2))
            spatial_kernel[i, j] = np.exp(-d ** 2 / (2 * sigma_space ** 2))

    # Implement the bilateral filter
    for i in range(height):
        for j in range(width):
            center_pixel = padded_image[i:i + k_size, j:j + k_size]
            color_kernel = np.exp(-(center_pixel - in_image[i, j]) ** 2 / (2 * sigma_color ** 2))
            bilateral_kernel = color_kernel * spatial_kernel

            # Normalize the kernel weights
            normalized_weights = bilateral_kernel / np.sum(bilateral_kernel)

            my_filtered_image[i, j] = np.sum(center_pixel * normalized_weights)

    return opencv_filtered_image, my_filtered_image.astype(np.uint8)
