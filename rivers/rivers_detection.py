from random import randint
import numpy as np
import cv2 as cv
from math import sqrt, sin, cos
import math


def rivers_detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thr = cv.adaptiveThreshold(
        gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        3,
        3
    )

    conts, hier = cv.findContours(
        thr,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_NONE
    )

    def hyp(x1, y1, x2, y2):
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def split_if_too_distant(to_split, r, scale=1.5):
        splitted = []
        current = []
        for p1, p2 in (zip(to_split[:-1], to_split[1:])):
            current.append(p1)
            if hyp(*tuple(p1[0]), *tuple(p2[0])) > r * scale:
                splitted.append(current[:])
                current = []

        current.append(to_split[-1])
        splitted.append(current)

        return splitted

    def clean_first(to_clean, r):
        filtered = []
        discarded = []

        x, y = to_clean.pop()[0]

        for point in to_clean:
            px, py = point[0]
            if hyp(x, y, px, py) > r:
                filtered.append(point)
            else:
                discarded.append(point)

        return x, y, filtered, discarded

    def reduce_contour(to_filter, r):
        filtered = []
        discarded = []

        while len(to_filter) > 0:
            x, y, to_filter, disc = clean_first(to_filter, r=r)
            filtered.append([[x, y]])
            discarded += disc

        return split_if_too_distant(filtered, r), discarded

    def contour_len(cont):
        l = 0
        for p1, p2 in (zip(cont[:-1], cont[1:])):
            l += hyp(*tuple(p1[0]), *tuple(p2[0]))
        return l

    reduced_contours = []
    for cont in conts:
        rconts, _ = reduce_contour(cont.tolist(), 5)
        reduced_contours += rconts

    def draw_contour(img, cont, color=(255, 255, 255)):
        for p1, p2 in (zip(cont[:-1], cont[1:])):
            cv.line(img, tuple(p1[0]), tuple(p2[0]), color, 1)

    # exc = np.zeros(img.shape, np.uint8)
    exc = np.copy(img)

    for cont in reduced_contours:
        if contour_len(cont) > 10:
            draw_contour(exc, cont)

    # color match

    # def drawline(img, point, color=(0, 255, 0)):
    #     cv.line(img, point, point, color, thickness=1)

    # longest = max(reduced_contours, key=len)
    # draw_contour(exc, longest)

    # center_index = int(len(longest) / 2)

    # a = longest[center_index - 1][0]
    # b = longest[center_index][0]
    # c = longest[center_index + 1][0]
    # print(a, b, c)

    # drawline(exc, (a[0], a[1]), color=(0, 255, 50))
    # drawline(exc, (b[0], b[1]), color=(0, 100, 255))
    # drawline(exc, (c[0], c[1]), color=(0, 255, 50))

    # y = 0
    # x = 1

    # m = (c[y] - a[y]) / (c[x] - a[x])
    # mb = - 1 / m

    # t = 12
    # alpha = math.atan(mb)
    # dy = sin(alpha) * t
    # dx = cos(alpha) * t
    # nx = int(b[x] + dx)
    # ny = int(b[y] + dy)
    # p = img[ny, nx]
    # color = tuple(map(int, (p[0], p[1], p[2])))
    # print(color)
    # drawline(exc, (ny, nx),  color=color)
    # print(dx, dy, hyp(b[x], b[y], nx, ny))

    # t = -t
    # alpha = math.atan(mb)
    # dy = sin(alpha) * t
    # dx = cos(alpha) * t
    # nx = int(b[x] + dx)
    # ny = int(b[y] + dy)
    # p = img[ny, nx]
    # color = tuple(map(int, (p[0], p[1], p[2])))
    # print(color)
    # drawline(exc, (ny, nx),  color=color)
    # print(nx, ny, hyp(b[x], b[y], nx, ny))

    cv.imwrite('out.jpg', exc)


#
#
#


def cut_image(img: np.ndarray, target_height_perc=65, target_width_perc=65) -> np.ndarray:
    """Cut an image in a rectangle of size defined by parameters, to remove black borders"""

    # retrieve image shape
    old_width, old_height, _ = img.shape
    half_width = old_width // 2
    half_height = old_height // 2

    # evaluate the needed padding on the x an y axis
    padding_x = (half_width * target_height_perc) // 100
    padding_y = (half_height * target_height_perc) // 100

    # cut the image
    return img[
        half_width - padding_x:half_width + padding_x,
        half_height - padding_y:half_height+padding_y
    ]


img = cv.imread('./data/46139916742_e461092961_o.jpg')
# img = cv.imread('./data/46139865612_61080b4f33_o.jpg')
img = cut_image(img)


rivers_detection(img)
