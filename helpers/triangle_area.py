import numpy as np


def area_triangle(l_sides):
    # calculate the semi-perimeter
    s = sum(l_sides) / 2

    # calculate the area
    area = np.prod([s, *[s - _ for _ in l_sides]]) ** 0.5

    return area
