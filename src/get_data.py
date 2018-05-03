import cv2
import numpy as np
import pandas as pd
from PIL import Image


out_path = r'D:\dataset\fer2013\fer2013.csv'
data = pd.read_csv(out_path, dtype='a')
label = np.array(data['emotion'])
img_data = np.array(data['pixels'])
usage = np.array(data['Usage'])

N_sample = label.size
train_num = len([n for n in usage if n == 'Training'])
test_num = len([n for n in usage if n == 'PublicTest'])
valid_num = len([n for n in usage if n == 'PrivateTest'])
n_input = 48*48  # data input (img shape: 48*48)
n_input_sliced = 42*42  # data input (img shape: 42*42)


def handle_array(x):
    x = np.asarray(x).flatten()
    x_max = x.max()
    x = x / (x_max + 0.0001)
    return x


def get_face_data_label_origin():
    """
    获取数据
    """
    f_data = np.zeros((N_sample, n_input))
    f_label = np.zeros(N_sample, dtype=int)

    for i in range(N_sample):
        x = img_data[i]
        x = np.fromstring(x, dtype=float, sep=' ').reshape(48, 48)
        f_data[i] = handle_array(x)
        f_label[i] = int(label[i])

    train_x = f_data[0:train_num, :]
    train_y = f_label[0:train_num]

    test_x = f_data[train_num:train_num+test_num, :]
    test_y = f_label[train_num:train_num+test_num]

    valid_x = f_data[train_num+test_num:train_num+test_num+valid_num, :]
    valid_y = f_label[train_num+test_num:train_num+test_num+valid_num]

    return train_x, train_y, test_x, test_y, valid_x, valid_y


def get_face_data_label_duplicated():
    """
    获取42*42的数据
    :return:
    """
    f_data = np.zeros((N_sample * 10, n_input_sliced))
    f_label = np.zeros(N_sample * 10, dtype=int)

    for i in range(N_sample):
        x = img_data[i]
        x = np.fromstring(x, dtype=float, sep=' ').reshape(48, 48)
        x1 = x[0:42, 0:42]
        f_data[10 * i + 0] = handle_array(x1)
        f_label[10 * i + 0] = int(label[i])
        img = cv2.flip(np.asarray(Image.fromarray(x1)), 1)
        f_data[10 * i + 1] = handle_array(img)
        f_label[10 * i + 1] = int(label[i])

        x1 = x[0:42, 6:48]
        f_data[10 * i + 2] = handle_array(x1)
        f_label[10 * i + 2] = int(label[i])
        img = cv2.flip(np.asarray(Image.fromarray(x1)), 1)
        f_data[10 * i + 3] = handle_array(img)
        f_label[10 * i + 3] = int(label[i])

        x1 = x[6:48, 0:42]
        f_data[10 * i + 4] = handle_array(x1)
        f_label[10 * i + 4] = int(label[i])
        img = cv2.flip(np.asarray(Image.fromarray(x1)), 1)
        f_data[10 * i + 5] = handle_array(img)
        f_label[10 * i + 5] = int(label[i])

        x1 = x[6:48, 6:48]
        f_data[10 * i + 6] = handle_array(x1)
        f_label[10 * i + 6] = int(label[i])
        img = cv2.flip(np.asarray(Image.fromarray(x1)), 1)
        f_data[10 * i + 7] = handle_array(img)
        f_label[10 * i + 7] = int(label[i])

        x1 = x[3:45, 3:45]
        f_data[10 * i + 8] = handle_array(x1)
        f_label[10 * i + 8] = int(label[i])
        img = cv2.flip(np.asarray(Image.fromarray(x1)), 1)
        f_data[10 * i + 9] = handle_array(img)
        f_label[10 * i + 9] = int(label[i])

    train_x = f_data[0:train_num * 10, :]
    train_y = f_label[0:train_num * 10]

    test_x = f_data[train_num * 10:(train_num+test_num) * 10, :]
    test_y = f_label[train_num * 10:(train_num+test_num) * 10]

    valid_x = f_data[(train_num+test_num) * 10:, :]
    valid_y = f_label[(train_num+test_num) * 10:]

    return train_x, train_y, test_x, test_y, valid_x, valid_y


def get_emotion_type():
    """
    获取情感分类
    :return:
    """
    return {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprised', 6: 'neutral'}
