import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 10,
})
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math as m
import sys
import argparse


def rgbToLab(p):
    w = [0.313, 0.329, 1 - .313 - .329]  # D65
    zn = 1.088754
    xn = 0.950456

    t = p[0] + p[1] + p[2]
    p[0] /= t  # B
    p[1] /= t  # G
    p[2] /= t  # R

    x = (.412453 * p[2] + .357580 * p[1] + .180423 * p[0]) / xn
    y = .2126710 * p[2] + .7151600 * p[1] + .0721690 * p[0]
    z = (.0193340 * p[2] + .1191930 * p[1] + .950227 * p[0]) / zn

    L = 0
    if y > 0.008856:
        L = 116 * (y ** (1. / 3)) - 16
    else:
        L = 903.3 * y

    L = 116 * (p[1] / w[1]) ** (1. / 3) - 16                            # lightness     : 0 black, 50 gray, 100 white
    a = 500 * ((p[2] / w[2]) ** (1. / 3) - (p[1] / w[1]) ** (1. / 3))   # red-green     : -100 green, 0 neutral, 100 red
    b = 200 * ((p[1] / w[1]) ** (1. / 3) - (p[0] / w[0]) ** (1. / 3))   # blue-yellow   : -100 blue, 0 neutral, 100 yellow

    return [L, a, b]


def imgValue(source):
    val = 0
    height = source.shape[0]
    width = source.shape[1]
    for i in range(height):
        for j in range(width):
            val += sum(source[i, j])
    return val


def imgValueMono(source):
    val = 0
    height = source.shape[0]
    width = source.shape[1]
    for i in range(height):
        for j in range(width):
            val += source[i, j]
        print(i)
    return val


def distanceSRGB(source, truth, c, total):
    dist = 0
    height = source.shape[0]
    width = source.shape[1]
    for i in range(height):
        for j in range(width):
            dist += m.sqrt((int(source[i, j][0]) - int(truth[i, j][0])) ** 2 + (
                    int(source[i, j][1]) - int(truth[i, j][1])) ** 2 + (
                                   int(source[i, j][2]) - int(truth[i, j][2])) ** 2)

    sys.stdout.write("\r" + str(c[0]) + "/" + str(total))
    sys.stdout.flush()
    c[0] = c[0] + 1

    return dist / (height * width)


def distanceLab(source, truth, c, total):
    dist = 0
    height = source.shape[0]
    width = source.shape[1]
    labSource = cv.cvtColor(source, cv.COLOR_RGB2Lab)

    for i in range(height):
        for j in range(width):
            dist += m.sqrt(
                (int(labSource[i, j][0]) - int(truth[i, j][0])) ** 2 + (
                            int(labSource[i, j][1]) - int(truth[i, j][1])) ** 2 + (
                        int(labSource[i, j][2]) - int(truth[i, j][2])) ** 2)

    sys.stdout.write("\r" + str(c[0]) + "/" + str(total))
    sys.stdout.flush()
    c[0] = c[0] + 1

    return dist / (height * width)


def distanceSRGBMono(source, truth, c, total):
    dist = 0
    height = source.shape[0]
    width = source.shape[1]
    for i in range(height):
        for j in range(width):
            dist += abs(int(source[i, j]) - int(truth[i, j]))

    sys.stdout.write("\r" + str(c[0]) + "/" + str(total))
    sys.stdout.flush()
    c[0] = c[0] + 1

    return dist / (height * width)


def calculateDistanceSeriesMono(imgs, truth, v):
    counter = [0]
    l = len(imgs)
    v = pd.Series([distanceSRGBMono(img[1], truth[1], counter, l) for img in imgs[:l]], index=[img[0] for img in imgs])


def showImage(img):
    cv.imshow('something', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # args handling
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", help="Output directory", type=str, nargs='?', default=r'/plots')
    parser.add_argument("-ld", "--listdir", help="List of input directories", type=str, nargs='+')
    # parser.add_argument("-re", "--realistic", help="Path of a realistic sample", type=str, default='')
    # parser.add_argument("-fit", "--fittizia", help="Path of an ideal sample", type=str, default='')

    args = parser.parse_args()
    # folders = [args.fittizia, args.realistic]
    folders = args.listdir
    print(folders)
    outputdir = args.output
    # fit = args.fittizia
    # re = args.realistic

    print(os.path.join(os.getcwd(), outputdir))
    if not os.path.isdir(os.path.join(os.getcwd(), outputdir)):
        os.makedirs(os.path.join(os.getcwd(), outputdir))

    i = 0
    distanceSeries = []
    # main cycle
    for folder in folders:

        # read input and prepare structure [index, image]
        imgTruth = [0, np.zeros((1080, 1920, 3), np.uint8)]                 # dummy blank image
        imgs = []

        for filename in os.listdir(folder):
            img = cv.imread(os.path.join(folder, filename))
            if img is not None:
                screenshotIndex = int(str(Path(filename).with_suffix('')))  # path(w/ no type) -> string -> int
                imgs.append([screenshotIndex, img])                         # big assertion here: filename has to be a number
                if imgTruth is None or screenshotIndex > imgTruth[0]:
                    imgTruth = [screenshotIndex, img]

        # sort images by index: the index represents time -> "sort by time"
        imgSeries = pd.Series([img[1] for img in imgs], index=[img[0] for img in imgs])
        imgSeries = imgSeries.sort_index()

        sys.stdout.write("testbed successfully read\n")
        sys.stdout.flush()
        uglyCounter = [1]
        tot = len(imgs)  # log progress

        labTruth = cv.cvtColor(imgTruth[1], cv.COLOR_RGB2Lab)

        sys.stdout.write("CIE L*a*b* ground truth computed \n")
        sys.stdout.flush()  # log progress

        # distances from ground truth
        distanceSeries.append(pd.Series([distanceLab(img[1], labTruth, uglyCounter, tot) for img in imgs],
                                      index=[img[0] for img in imgs]))
        distanceSeries[i] = distanceSeries[i].sort_index()

        # partial output
        sys.stdout.write(folder + " done\n")
        sys.stdout.write('Area: ' + str(distanceSeries[i].sum()) + " \n")
        sys.stdout.flush()
        i += 1

    # if distanceSeries[0].size < distanceSeries[1].size:
    #     distanceSeries[0] = distanceSeries[0].append(pd.Series(np.zeros(distanceSeries[1].size - distanceSeries[0].size, dtype=np.uint8)), ignore_index=True)

    plt.figure(figsize=(5.9, 3.4), tight_layout=True)
    plt.xlabel("Tempo (Frame)")
    plt.ylabel("Differenza percettiva")
    plt.plot(distanceSeries[0], '-', linewidth=2, label='Scena realistica')
    # plt.plot(distanceSeries[1], '-', linewidth=2, label='Scena realistica')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(os.getcwd(), outputdir, os.path.basename(os.path.normpath(folder)) + " " + str(round(distanceSeries[0].sum(), 2)) + ".pgf"))
# + "-" + str(round(distanceSeries[1].sum(), 2))
