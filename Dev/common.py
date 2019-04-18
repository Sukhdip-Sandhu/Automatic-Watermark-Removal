from matplotlib import pyplot as plt

DEBUG_MODE = True


def debug(function, *args, **kwargs):
    if DEBUG_MODE:
        function(*args, **kwargs)


def show_image(image):
    plt.imshow(image, cmap='gray')
