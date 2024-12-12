import numpy as np
import cv2

def pretty_depth(depth):
    """Converts depth into a 'nicer' format for display

    This is abstracted to allow for experimentation with normalization

    Args:
        depth: A numpy array with 2 bytes per pixel

    Returns:
        A numpy array that has been processed whose datatype is unspecified
    """
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

def pretty_depth_cv(depth):
    """Converts depth into a 'nicer' format for display

    This is abstracted to allow for experimentation with normalization

    Args:
        depth: A numpy array with 2 bytes per pixel

    Returns:
        An OpenCV image (numpy array) that can be displayed
    """
    depth = pretty_depth(depth)
    # In cv2, we can directly return the numpy array as it is compatible
    return depth

def video_cv(video):
    """Converts video into a BGR format for OpenCV

    This is abstracted out to allow for experimentation

    Args:
        video: A numpy array with 1 byte per pixel, 3 channels RGB

    Returns:
        An OpenCV image (numpy array) who's datatype is 1 byte, 3 channel BGR
    """
    # Convert RGB to BGR
    video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
    return video
