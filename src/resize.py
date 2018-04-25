import os
import cv2


test_path = "D:\\dataset\\fer2013\out\\train"
out_path = "D:\\dataset\\fer2013\\train"

for d in os.listdir(test_path):
    d_path = os.path.join(test_path, d)
    o_path = os.path.join(out_path, d)
    os.mkdir(o_path)
    for _, _, files in os.walk(d_path):
        for img in files:
            img_path = os.path.join(d_path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (42, 42))
            f = os.path.join(o_path, img)
            cv2.imwrite(f, image)
