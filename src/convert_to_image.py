import csv
import os
from PIL import Image
import numpy as np


# 原始的csv
file_path = r'D:\dataset\fer2013\fer2013.csv'
# 删除无效训练集后的csv
out_path = r'D:\dataset\fer2013\fer2013_.csv'
datasets_path = r'D:\dataset\fer2013\out'
database_path = r'D:\dataset\fer2013'
train_csv = os.path.join(datasets_path, 'train.csv')
val_csv = os.path.join(datasets_path, 'val.csv')
test_csv = os.path.join(datasets_path, 'test.csv')

train_set = os.path.join(datasets_path, 'train')
val_set = os.path.join(datasets_path, 'val')
test_set = os.path.join(datasets_path, 'test')


def remove_bad():
    """
    删除无效训练集
    """
    # 无效训练集所在的行号
    bad_train_path = r'..\badtrainingdata.txt'
    exclude_index = []

    with open(bad_train_path) as f:
        rows = f.readlines()
        for row in rows:
            exclude_index.append(int(row) - 2)
    """
    [
        59, 2059, 2809, 3262, 3931, 4275, 5274, 5439, 5722, 5881,
        6102, 6458, 7172, 7496, 7527, 7629, 8030, 8737, 8856, 9026,
        9500, 9680, 10423, 11244, 11286, 11295, 11846, 12289, 12352, 13148,
        13402, 13988, 14279, 15144, 15838, 15894, 17081, 19238, 19632, 20222,
        20712, 20817, 21817, 22198, 22927, 23596, 23894, 24441, 24891, 25219,
        25603, 25909, 26383, 26860, 26897, 28601, 29073, 29094, 30745, 31532,
        31880, 32043, 32154, 33036, 33146, 33591, 34294, 34570]
    """

    exclude_index.reverse()

    with open(file_path) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        rows = [row for row in csvr]

        trn = [row for row in rows]
        for index in exclude_index:
            trn.pop(index)

        csv.writer(open(out_path, 'w+'), lineterminator='\n').writerows([header] + trn)
        print(len(trn))


def split_csv():
    os.mkdir(datasets_path)

    with open(out_path) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        rows = [row for row in csvr]

        trn = [row[:-1] for row in rows if row[-1] == 'Training']
        csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
        print(len(trn))

        val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
        csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
        print(len(val))

        tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
        print(len(tst))


def write_image():
    num = 0
    for save_path, csv_file in [(train_set, train_csv), (test_set, test_csv), (val_set, val_csv)]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(csv_file) as f:
            csvr = csv.reader(f)
            next(csvr)
            for i, (label, pixel) in enumerate(csvr):
                pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                subfolder = os.path.join(save_path, label)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                im = Image.fromarray(pixel).convert('L')
                image_name = os.path.join(subfolder, '{:05d}.jpg'.format(num))
                print(image_name)
                im.save(image_name)
                num += 1


remove_bad()
split_csv()
write_image()
