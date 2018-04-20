import cv2
import os


train_base = "D:\\dataset\\fer2013\\out\\train"
multi_base = "D:\\dataset\\fer2013\\multi"

for d in os.listdir(train_base):
    out = os.path.join(multi_base, d)
    if not os.path.exists(out):
        os.mkdir(out)
    label_dir = os.path.join(train_base, d)

    for img in os.listdir(label_dir):
        image = cv2.imread(os.path.join(label_dir, img))
        img_ext = os.path.splitext(img)

        # 左上
        out_pic = os.path.join(out, "{}{}".format(img_ext[0] + "_1", img_ext[1]))
        cv2.imwrite(out_pic, image[0:42, 0:42])

        # 右上
        out_pic = os.path.join(out, "{}{}".format(img_ext[0] + "_2", img_ext[1]))
        cv2.imwrite(out_pic, image[0:42, 6:48])

        # 左下
        out_pic = os.path.join(out, "{}{}".format(img_ext[0] + "_3", img_ext[1]))
        cv2.imwrite(out_pic, image[6:48, 0:42])

        # 右下
        out_pic = os.path.join(out, "{}{}".format(img_ext[0] + "_4", img_ext[1]))
        cv2.imwrite(out_pic, image[6:48, 6:48])

        # 正中
        out_pic = os.path.join(out, "{}{}".format(img_ext[0] + "_5", img_ext[1]))
        cv2.imwrite(out_pic, image[3:45, 3:45])

        # 缩放
        out_pic = os.path.join(out, "{}{}".format(img_ext[0] + "_6", img_ext[1]))
        image = cv2.resize(image, (42, 42))
        cv2.imwrite(out_pic, image)

        print(img)

for d in os.listdir(multi_base):
    label_dir = os.path.join(multi_base, d)

    for img in os.listdir(label_dir):
        image = cv2.imread(os.path.join(label_dir, img))
        img_ext = os.path.splitext(img)
        image = cv2.flip(image, 1)
        out_pic = os.path.join(label_dir, "{}{}".format(img_ext[0] + "_1", img_ext[1]))
        cv2.imwrite(out_pic, image)

        print(img)
