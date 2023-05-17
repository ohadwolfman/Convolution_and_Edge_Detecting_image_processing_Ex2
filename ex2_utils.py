import math
import numpy as np
import cv2

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
    return np.convolve(in_signal,k_size,"full")


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
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    derivKernel = np.array([1,0,-1])
    derivX = conv2D(in_image,derivKernel.transpose())
    derivY = conv2D(in_image,derivKernel)

    magnitude = np.sqrt((derivX**2)+(derivY**2))
    direction = np.arctan2(derivY,derivX)
    return direction, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
