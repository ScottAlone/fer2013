import os
import csv
import scipy.misc
import scipy
import cv2
import numpy as np
import imutils


def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector


def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T) / inputs.shape[1]  # Correlation matrix
    U, S, V = np.linalg.svd(sigma)  # Singular Value Decomposition
    epsilon = 0.1  # Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)  # Data whitening


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True,
                              sqrt_bias=10, min_divisor=1e-8):
    """
    __author__ = "David Warde-Farley"
    __copyright__ = "Copyright 2012, Universite de Montreal"
    __credits__ = ["David Warde-Farley"]
    __license__ = "3-clause BSD"
    __email__ = "wardefar@iro"
    __maintainer__ = "David Warde-Farley"
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]
    else:
        X = X.copy()
    if use_std:
        ddof = 1
        if X.shape[1] == 1:
            ddof = 0
        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X


def ZeroCenter(data):
    data = data - np.mean(data, axis=0)
    return data


def normalize(arr):
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval))
    return arr


def Flip(data):
    dataFlipped = data[..., ::-1].reshape(2304).tolist()
    return dataFlipped


def Roated15Left(data):
    num_rows, num_cols = data.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 30, 1)
    img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
    return img_rotation.reshape(2304).tolist()


def Roated15Right(data):
    num_rows, num_cols = data.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), -30, 1)
    img_rotation = cv2.warpAffine(data, rotation_matrix, (num_cols, num_rows))
    return img_rotation.reshape(2304).tolist()


def Zoomed(data):
    datazoomed = scipy.misc.imresize(data, (60, 60))
    datazoomed = datazoomed[5:53, 5:53]
    datazoomed = datazoomed.reshape(2304).tolist()
    return datazoomed


def shiftedUp20(data):
    translated = imutils.translate(data, 0, -5)
    translated2 = translated.reshape(2304).tolist()
    return translated2


def shiftedDown20(data):
    translated = imutils.translate(data, 0, 5)
    translated2 = translated.reshape(2304).tolist()
    return translated2


def shiftedLeft20(data):
    translated = imutils.translate(data, -5, 0)
    translated2 = translated.reshape(2304).tolist()
    return translated2


def shiftedRight20(data):
    translated = imutils.translate(data, 5, 0)
    translated2 = translated.reshape(2304).tolist()
    return translated2


def outputImage(pixels, number):
    data = pixels
    name = str(number) + "output.jpg"
    scipy.misc.imsave(name, data)


def Zerocenter_ZCA_whitening_Global_Contrast_Normalize(list):
    Intonumpyarray = np.asarray(list)
    data = Intonumpyarray.reshape(48, 48)
    data2 = ZeroCenter(data)
    data3 = zca_whitening(flatten_matrix(data2)).reshape(48, 48)
    data4 = global_contrast_normalize(data3)
    data5 = np.rot90(data4, 3)
    return data5


def read_single_image(path):
    """
    获取单个图片
    :param path: 图片路径
    :return: 文件名，图片信息
    """
    pixels = np.asarray(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (48, 48))).flatten()
    temp_list = []
    for pixel in pixels:
        temp_list.append(int(pixel))

    return [os.path.basename(path)], [Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)]


def read_image(path):
    """
    :param path: 图片路径
    :return:  文件名，图片信息
    """
    pixels = np.asarray(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (48, 48))).flatten()
    temp_list = []
    for pixel in pixels:
        temp_list.append(int(pixel))

    return Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)


def read_folder(path):
    """
    获取某个文件夹下所有图片
    :param path: 文件夹
    :return: 文件名，图片信息
    """
    file_names = []
    test_set_x = []
    for _, _, files in os.walk(path):
        for f in files:
            if f.endswith("png"):
                file_names.append(f)
                test_set_x.append(read_image(os.path.join(path, f)))

    return file_names, test_set_x


def load_test_data():
    f = open(r'C:\Users\Scott\Desktop\report\report6\fer2013.csv')
    csv_f = csv.reader(f)
    test_set_x = []
    test_set_y = []
    for row in csv_f:
        if str(row[2]) == "PrivateTest":
            test_set_y.append(int(row[0]))
            temp_list = []
            for pixel in row[1].split():
                temp_list.append(int(pixel))
            data = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)
            test_set_x.append(data)
    return test_set_x, test_set_y


def load_data():
    train_x = []
    train_y = []
    val_x = []
    val_y = []

    with open(r"..\badtrainingdata.txt", "r") as text:
        ToBeRemovedTrainingData = []
        for line in text:
            ToBeRemovedTrainingData.append(int(line))
    number = 0

    f = open(r'C:\Users\Scott\Desktop\report\report6\fer2013.csv')
    csv_f = csv.reader(f)

    for row in csv_f:
        number += 1
        if number not in ToBeRemovedTrainingData:

            if str(row[2]) == "Training" or str(row[2]) == "PublicTest":
                temp_list = []

                for pixel in row[1].split():
                    temp_list.append(int(pixel))

                data = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)
                train_y.append(int(row[0]))
                train_x.append(data.reshape(2304).tolist())

            elif str(row[2]) == "PrivateTest":
                temp_list = []

                for pixel in row[1].split():
                    temp_list.append(int(pixel))

                data = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(temp_list)
                val_y.append(int(row[0]))
                val_x.append(data.reshape(2304).tolist())

    return train_x, train_y, val_x, val_y
