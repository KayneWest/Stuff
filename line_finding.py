# Python OCR

import sys
import Image
import itertools
import numpy as np

  
from skimage.transform import probabilistic_hough_line
from skimage.filter import canny, sobel
#from skimage import data

#import matplotlib.pyplot as plt

# Line finding, using the Probabilistic Hough Transform

import Image
import ImageDraw


if sys.version_info.major < 3:
    range = xrange 

# functions

def image_to_array(im, reshape=True):
    """
    Converts a PIL Image object into a numpy array representing
    a greyscale image.
    Parameters:
        im - a PIL Image object
        reshape (default: True) - if True then returns a 2D image
                                otherwise it returns a flat 1D araray.
    Output:
        A numpy array.
    """
    im_greyscale = im.convert("L")
    width, height = im_greyscale.size
    im_array = np.ndarray(width * height)
    for i, p in enumerate(im_greyscale.getdata()):
        im_array[i] = p
    if reshape:
        im_array.shape = (height, width)
    return im_array


def array_to_image(array):
    height, width = array.shape
    im = Image.new('L', (width, height), 255)
    for x in range(width):
        for y in range(height):
            im.putpixel((x,y), array[y][x])
    return im


def windows0(im, rectangles, output_size):
    im_width, im_height = im.size
    for dx, dy in rectangles:
        for x0, y0 in itertools.product(range(im_width-dx+1),range(im_height-dy+1)):
            yield im.crop((x0, y0, x0+dx, y0+dy))


def windows(im, rectangles, output_size):
    """
    Parameters:
    im - a PIL image
    rectangles - a list of (width, height) pairs of rectangle sizes
    output_size - the desired output size in (width, height) format
    """
    im_width, im_height = im.size
    for dx, dy in rectangles:
        for x0, y0 in itertools.product(range(im_width-dx+1),range(im_height-dy+1)):
            yield im.transform(output_size, Image.EXTENT, (x0, y0, x0+dx, y0+dy))


def array_windows(im, rectangles, output_size):
    """
    Parameters:
    im - a PIL image
    rectangles - a list of (width, height) pairs of rectangle sizes
    output_size - the desired output size in (width, height) format
    """
    return (image_to_array(w) for w in windows(im, rectangles, output_size))



def draw_lines(lines, shape):
    """
    Find clusters of lines.
    """
    im = Image.new('L', shape, 255)
    draw = ImageDraw.Draw(im)
    for (x0, y0), (x1, y1) in lines:
        draw.line([(x0, y0), (x1, y1)], width = 1)
    return im


def avg_y(l):
    (x0, y0), (x1, y1) = l
    return (y0 + y1) / 2


def max_y(l):
    (x0, y0), (x1, y1) = l
    return max(y0, y1)


def min_y(l):
    (x0, y0), (x1, y1) = l
    return min(y0, y1)


def cmp_lines(l0, l1):
    if avg_y(l0) < avg_y(l1):
        return -1
    else:
        return 1


def regions(lines, border):
    regions = []
    current = ((0,0), (0,0))
    for l in lines:
        if len(regions) == 0:
            regions.append([l])
            current = ((0, max_y(l)), (0, max_y(l)))
        elif min_y(l) < max_y(current) + border:
            regions[-1].append(l)
            current = ((0, max(max_y(l), max_y(current))), (0, max(max_y(l), max_y(current))))
        else:
            regions.append([l])
            current = ((0, max_y(l)), (0, max_y(l)))
    return regions


def bounding_rectangle(lines, border):
    rx0 = min([x0 for (x0, y0), (x1, y1) in lines])
    ry0 = min([y0 for (x0, y0), (x1, y1) in lines])
    rx1 = max([x1 for (x0, y0), (x1, y1) in lines])
    ry1 = max([y1 for (x0, y0), (x1, y1) in lines])
    return (rx0-border, ry0-border), (rx1+border, ry1+border)


def text_sections(im, output_height):
    im = im.convert('L')
    im_array = image_to_array(im)
    im_arary = im_array / 255

    r = range(-50,50)
    r_theta = [np.pi/2 + x*0.001 for x in r]
    theta = np.array(r_theta)

    edges = canny(im_array, 2, 1, 25)

    lines = probabilistic_hough(edges, threshold=5, line_length=20,
                                line_gap=20, theta=theta)

    lines_sorted = sorted(lines, cmp=cmp_lines)
    rs = regions(lines_sorted, 2)
    text_areas = [bounding_rectangle(ls, 4) for ls in rs]
    for (x0, y0), (x1, y1) in text_areas:
        width, height = x1 - x0, y1 - y0
        output_width = width * output_height // height
        yield(im.transform((output_width, output_height), Image.EXTENT,
                           (x0, y0, x1, y1)))

if __name__ == '__main__':
    # testing code
    testfile = "data/hk.png"
    im_raw = Image.open(testfile)
    im_array = image_to_array(im_raw)
if __name__ == '__main__':
    im = Image.open('data/hk_eng_crop_2.png')
    text_sections(im, 20)
    for t in text_sections(im, 20):
        t.show()    
  
