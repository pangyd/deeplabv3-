import cv2 as cv
from PIL import Image
import os
import numpy as np
import random
from torchvision import transforms

# h, w
h, w = 640, 640

# 设置一张640*640的灰度图
img_gray = Image.new("RGB", (h, w), (128, 128, 128))
img_ = np.array(img_gray)
# cv.imshow("img_", img_)

"""
将灰度图划分成四个区域，确定中心点
district 1:(center_w, center_h)
district 2:(w - center_w, center_h)
district 3:(center_w, h - center_h)
district 4:(w - center_w, h - center_h)
"""
center = random.uniform(0.4, 0.6)
center_h = int(h * center)
center_w = int(w * center)
district_size = [(center_w, center_h), (w - center_w, center_h), (center_w, h - center_h), (w - center_w, h - center_h)]
xy_list = [(0, 0), (center_w, 0), (0, center_h), (center_w, center_h)]

# 获取原数据
path = "img"
img_list = os.listdir(path)[:4]
print(img_list)
paste_img_list = []
# 随机取四张
for i, district, xy in zip(range(4), district_size, xy_list):
    random_num = random.randint(0, len(img_list))

    # img = cv.imread(os.path.join(path, img_list[random_num]))
    img = cv.imread(os.path.join(path, img_list[i]))

    # 随机缩放
    img_h, img_w = img.shape[0], img.shape[1]
    random_size = random.uniform(1.3, 1.7)
    img = cv.resize(img, (int(img_w / random_size), int(img_h / random_size)))

    # 随机裁剪
    img_h, img_w = img.shape[0], img.shape[1]
    top = random.uniform(0.1, 0.3)
    bottom = random.uniform(0.7, 0.9)
    top_w, top_h = int(img_w * top), int(img_h * top)
    bottom_w, bottom_h = int(img_w * bottom), int(img_h * bottom)
    img = img[top_h: bottom_h, top_w: bottom_w, :]

    # paste_img_list.append(img_list[random_num])
    h, w = img.shape[0], img.shape[1]

    if (h > district[0]) & (w > district[1]):
        img = img[: district[0], : district[1], :]
    elif (h > district[0]) & (w <= district[1]):
        img = img[: district[0], :, :]
        img = cv.resize(img, (district[0], district[1]))
    elif (h <= district[0]) & (w > district[1]):
        img = img[:, : district[1], :]
        img = cv.resize(img, (district[0], district[1]))
    else:
        img = cv.resize(img, (district[0], district[1]))

    # ndarray --> PIL
    img_new = Image.fromarray(img.astype("uint8")).convert("RGB")
    # paste
    # print(xy)
    img_gray.paste(img_new, xy)

# PIL --> ndarray
img_cv = np.array(img_gray)
cv.imshow("mosaic", img_cv)
cv.waitKey(0)
cv.destroyAllWindows()


