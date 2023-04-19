import os
import shutil
import cv2 as cv
from PIL import Image
import numpy as np


def remove_img(path):
    img_list = os.listdir(path)

    for img in img_list:
        if "pyr" in img:
            os.remove(os.path.join(path, img))


def json2png(json_folder, png_save_folder):
    if os.path.isdir(png_save_folder):
        shutil.rmtree(png_save_folder)
    os.makedirs(png_save_folder)

    json_files = os.listdir(json_folder)
    for json_file in json_files:
        # 只遍历json文件，跳过文件夹
        # if not os.path.isdir(os.path.join(json_folder, json_file)):
        if os.path.isfile(os.path.join(json_folder, json_file)):
            print(json_file)
            json_path = os.path.join(json_folder, json_file)
            os.system("labelme_json_to_dataset {}".format(json_path))   # 修改单个json文件

            label_path = os.path.join(json_folder, json_file.split(".")[0] + "_json/label.png")   # 存放在新文件夹下

            png_save_path = os.path.join(png_save_folder, json_file.split(".")[0] + ".png")
            label_png = cv.imread(label_path, 0)
            label_png[label_png > 0] = 255
            cv.imwrite(png_save_path, label_png)

            shutil.rmtree(os.path.join(json_folder, json_file.split(".")[0] + "_json"))


def move_data2train():
    origin_img = "datasets/smoke/smoke_image"
    # origin_label = "datasets/smoke/smoke_label"
    new_img = "D://deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/JPEGImages"
    # new_label = "D://deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/SegmentationClass"

    if not os.path.exists(new_img):
        os.makedirs(new_img)
    # if not os.path.exists(new_label):
    #     os.makedirs(new_label)

    img_list = os.listdir(origin_img)
    # label_list = os.listdir(origin_label)
    move_img = os.listdir(new_img)
    # move_label = os.listdir(new_label)

    # for img, label in zip(img_list, label_list):
    for img in img_list:
        if img not in move_img:
            ori_img_path = os.path.join(origin_img, img)
            new_img_path = os.path.join(new_img, img)
            shutil.copy(ori_img_path, new_img_path)
        # if label not in move_label:
        #     ori_label_path = os.path.join(origin_label, label)
        #     new_label_path = os.path.join(new_label, label)
        #     shutil.copy(ori_label_path, new_label_path)


def move_img(origin_path, new_path, label_path):
    json_file = os.listdir(label_path)
    json_list = [i.split(".")[0] for i in json_file]

    origin_img_list = os.listdir(origin_path)
    for origin_img in origin_img_list:
        if origin_img.split(".")[0] in json_list:
            old_path = os.path.join(origin_path, origin_img)
            # new_path = os.path.join(new_path, origin_img)
            shutil.copy(old_path, new_path)


def alter_label():
    """三通道改成一通道"""
    label_root = "datasets/smoke/smoke_label_1"
    label_list = os.listdir(label_root)
    for label in label_list:
        label_path = os.path.join(label_root, label)
        label_img = Image.open(label_path)
        label_img = label_img.convert("L")
        label_cv = np.array(label_img)
        print(label_cv.shape)
        cv.imwrite(label_path, label_cv)


def alter_label_val():
    label_folder = "datasets/smoke/smoke_label"
    save_path = "datasets/smoke/smoke_label_1"
    label_list = os.listdir(label_folder)
    save_list = os.listdir(save_path)
    for l in save_list:
        # shutil.rmtree(os.path.join(save_path, l))
        os.remove(os.path.join(save_path, l))

    for label in label_list:
        label_path = os.path.join(label_folder, label)
        label_img = cv.imread(label_path)
        # 修改像素值：像素值为类别
        label_img[label_img[:, :, :] > 0] = 1
        cv.imwrite(os.path.join(save_path, label), label_img)


if __name__ == "__main__":
    img_path = "datasets/smoke/smoke_image"
    remove_img(img_path)

    # move_file()

    # json_folder = "D://yolov5-pytorch-main/smoke/smoke_json_new"
    # png_save_folder = "datasets/smoke/smoke_label"
    # json2png(json_folder=json_folder, png_save_folder=png_save_folder)

    # move_img("D://yolov5-pytorch-main/smoke/images", "datasets/smoke/smoke_image", "datasets/smoke/smoke_label")
    # alter_label()








