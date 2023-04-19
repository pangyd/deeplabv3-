import os
import random
import shutil

import cv2 as cv
from PIL import Image
import numpy as np
import warnings
from scipy import misc
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")


# img_path = "img/test.jpg"
# input_size = [640, 640]
# box = np.array([np.array([320, 370, 540, 570])])
# print(box)

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def boost_img(img_path, input_size, box, jitter=0.3, hu=0.3, s=0.7, v=0.4, random=True):
    img = Image.open(img_path)
    img = img.convert("RGB")

    iw, ih = img.size
    w, h = input_size[0], input_size[1]
    # print(iw, ih, w, h)

    if not random:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        # 给修改的图像加上灰条
        img = img.resize((nw, nh), Image.BICUBIC)
        new_img = Image.new("RGB", (w, h), (128, 128, 128))
        new_img.paste(img, (dx, dy))
        img_data = np.array(new_img, np.float32)

        # 对真实框进行调整
        if len(box) > 0:
            # np.random.shuffle()
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box_new = box[np.logical_and(box_w > 1, box_h > 1)]
        return img_data, box_new

    # 对图像进行缩放
    rs = iw / ih * rand(1-jitter, 1+jitter) / rand(1-jitter, 1+jitter)
    scaler = rand(0.25, 2)
    if rs < 1:
        nh = int(h * scaler)
        nw = int(nh * rs)
    else:
        nw = int(w * scaler)
        nh = int(nw / rs)
    img = img.resize((nw, nh), Image.BICUBIC)

    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    img_new = Image.new("RGB", (w, h), (128, 128, 128))
    img_new.paste(img, (dx, dy))
    img = img_new
    print(img)

    # 翻转图像
    flip = rand() < 0.5
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_data = np.array(img, np.uint8)   # cv处理数据为ndarray数据格式
    dtype = img_data.dtype

    # 对图像的色域进行变换 -- HSV -- 可增加亮度
    r = np.random.uniform(-1, 1, 3) * [hu, s, v] + 1
    hu, s, v = cv.split(cv.cvtColor(img_data, cv.COLOR_RGB2HSV))

    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)

    # 对修改后的HSV应用
    img_data = cv.merge((cv.LUT(hu, lut_h), cv.LUT(s, lut_s), cv.LUT(v, lut_v)))
    img_data = cv.cvtColor(img_data, cv.COLOR_HSV2RGB)

    if len(box) > 0:
        # np.random.shuffle()
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box_new = box[np.logical_and(box_w > 1, box_h > 1)]

    return img_data, box_new


# 水平翻转
def horizontal_flip(img_path, box):
    img = cv.imread(img_path)
    if random.random() < 0.5:
        h, w, c = img.shape()
        img = img[:, ::-1, :]   # w维度翻转[0:-1:-1]
        box[:, [0, 2]] = w - box[:, [2, 0]]
    return img, box


# 竖直翻转
def vertical_flip(img_path, box):
    img = cv.imread(img_path)
    if random.random() < 0.5:
        h, w, c = img.shape()
        img = img[::-1, :, :]
        box[:, [1, 3]] = h - box[:, [3, 1]]
    return img, box


# 旋转90度
def random_rot90(img_path, box):
    img = cv.imread(img_path)
    # cv.imshow("ori", img)
    if random.random() < 0.5:
        h, w, c = img.shape
        img_trans = cv.transpose(img)   # 矩阵转置
        # cv.imshow("new", img_trans)
        img_new = cv.flip(img_trans, 1)   # 顺时针旋转
        # img_new = cv.flip(img_trans, 0)   # 逆时针旋转

        cv.waitKey(0)
        cv.destroyAllWindows()
        if box is None:
            return img_new
        else:
            box[:, [0, 1, 2, 3]] = box[:, [1, 0, 3, 2]]
            box[:, [0, 2]] = h - box[:, [0, 2]]   # 顺时针
            # box[:, [1, 3]] = w - box[:, [1, 3]]   # 逆时针
            return img_new, box
    else:
        if box is None:
            return img
        else:
            return img, box


def rotate90(img_path):
    img = cv.imread(img_path)
    img = img[:500, :400, :]
    cv.imshow("ori", img)
    print(img.shape)
    h, w = img.shape[:2]
    """"""
    rotate_metrix = cv.getRotationMatrix2D((w/2, h/2), 90, 1)
    """"""
    img = cv.warpAffine(img, rotate_metrix, (w, h))   # (w, h):画布大小
    print(img.shape)
    cv.imshow("new", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img


def lv_move(img, move_size):
    dx = round(random.uniform(-move_size, move_size))
    dy = round(random.uniform(-move_size, move_size))
    h, w, c = img.shape

    y1 = move_size + 1 + dy
    y2 = y1 + h
    x1 = move_size + 1 + dx
    x2 = x1 + w

    makeborder = cv.copyMakeBorder(img, move_size+1, move_size+1, move_size+1, move_size+1, borderType=cv.BORDER_REFLECT101)
    img = makeborder[y1:y2, x1:x2, :]
    return img


def cutout(img, max_h_size=8, max_w_size=8):
    h = img.shape[0]
    w = img.shape[1]
    print(w, h)

    for _ in range(50):
        y = np.random.randint(h)
        x = np.random.randint(w)
        print(x, y)

        y1 = np.clip(max(0, y - max_h_size // 2), 0, h)
        y2 = np.clip(max(0, y + max_h_size // 2), 0, h)
        x1 = np.clip(max(0, x - max_w_size // 2), 0, w)
        x2 = np.clip(max(0, x + max_w_size // 2), 0, w)
        print(y1, y2, x1, x2)
        img[y1:y2, x1:x2, :] = 0
    return img


def rescale(img, output_size):
    raw_h, raw_w = img.shape[:2]

    img_new = cv.resize(img, (output_size, output_size))
    h, w = img_new.shape[:2]

    if w > raw_w:
        i = np.random.randint(0, h - raw_h)
        j = np.random.randint(0, w - raw_w)
        img_new = img_new[i:i+raw_h, j:j+raw_w, :]
    else:
        res_w = raw_w - w
        img_new = cv.copyMakeBorder(img_new, res_w, 0, res_w, 0, cv.BORDER_REFLECT)
    return img_new

img = cv.imread("datasets/smoke/smoke_image/smoke_00001.jpg")
cv.imshow("origin", img)
# img_new = lv_move(img, 50)
# img_new = cutout(img)
img_new = rescale(img, 400)
cv.imshow("new", img_new)
img_new = rescale(img, 500)
cv.imshow("new2", img_new)
cv.waitKey(0)
cv.destroyAllWindows()

# 仿射变换
def affine_changed(img_path):
    img = cv.imread(img_path)
    img = img[:500, :400, :]
    cv.imshow("ori", img)
    print(img.shape)
    h, w = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    print(pts1.shape)
    pts2 = np.float32([[100, 100], [200, 50], [100, 250]])
    """"""
    affine_matrix = cv.getAffineTransform(pts1, pts2)
    """"""
    img = cv.warpAffine(img, affine_matrix, (w, h))
    cv.imshow("new", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img


def pyramid_changed(img_path, img_name, img_type):
    """获取图像中有意义的信息，而一张图像中含有不同尺寸下的有意义的信息"""
    img = cv.imread(os.path.join(img_path, img_name))
    # img = img[:500, :400, :]
    # cv.imshow("ori", img)

    img_down = cv.pyrDown(img)   # 高斯平滑 -> 降采样
    img_up = cv.pyrUp(img)   # 升采样 -> 高斯平滑

    # 生成数据
    cv.imwrite("{}/{}_pyrdown.{}".format(img_path, img_name.split(".")[0], img_type), img_down)
    cv.imwrite("{}/{}_pyrup.{}".format(img_path, img_name.split(".")[0], img_type), img_up)


# 生成金字塔数据
# img_root = "./datasets/smoke/smoke_image"
# label_root = "./datasets/smoke/smoke_label"
# img_list = os.listdir(img_root)
# label_list = os.listdir(label_root)
# for img, label in zip(img_list, label_list):
#     pyramid_changed(img_root, img, "jpg")
    # pyramid_changed(label_root, label, "png")

# for img in img_list:
#     if "pyr" in img:
#         os.remove(os.path.join(img_root, img))
# for label in label_list:
#     if "pyr" in label:
#         os.remove(os.path.join(label_root, label))



# 随机转换通道
def random_change_channels(img_path):
    img = cv.imread(img_path)
    bgr = {0: "B", 1: "G", 2: "R"}
    channels = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    if random.random() < 0.5:
        changed_channels = channels[random.randrange(0, len(channels))]
        print("转换后的通道为：{}{}{}".format(bgr[changed_channels[0]], bgr[changed_channels[1]], bgr[changed_channels[2]]))
        img[:, :, (0, 1, 2)] = img[:, :, changed_channels]
    return img


# 随机修改饱和度
def random_saturate(img_path, lower=0.5, upper=1.5):
    img = cv.imread(img_path)
    if random.random() < 0.5:
        img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(lower, upper))
    return img


# 随机翻转0~90度
def random_rotate(img_path, lower, upper):
    img = cv.imread(img_path)
    cv.imshow("ori", img)
    angle = random.uniform(lower, upper)
    img = misc.imrotate(img, angle, "bicubic")
    cv.imshow("new", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img


def readAnnotation(xml_path):
    et = ET.parse(xml_path)
    annota = et.getroot()
    objs = annota.findall("object")
    x_ys = []
    for obj in objs:
        x_y = []
        label_name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = int(round(float(bnd.find("xmin").text)))
        ymin = int(round(float(bnd.find("ymin").text)))
        xmax = int(round(float(bnd.find("xmax").text)))
        ymax = int(round(float(bnd.find("ymax").text)))

        x_y.append(xmin)
        x_y.append(ymin)
        x_y.append(xmax)
        x_y.append(ymax)
        x_y.append(0)

        x_ys.append(x_y)
    return x_ys


# img_data, box = boost_img(img_path, input_size)
# img = Image.fromarray(img_data.astype(np.uint8))

# img.show()


# et = ET.parse("VOCdevkit/VOC2007/Annotations/144.xml")
# annota = et.getroot()
# objs = annota.findall("object")
# for obj in objs:
#     print(obj)


