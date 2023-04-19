import cv2
import os
from PIL import Image

# output_dir = "img_out"
# output_list = os.listdir(output_dir)
#
# for output in output_list[:2]:
#     img = cv.imread(os.path.join(output_dir, output))
#     h, w, c = img.shape[0], img.shape[1], img.shape[2]
#
#     for j in range(w):
#         for k in range(h):
#             if (img[k, j, 0] != 100) & (img[k, j, 0] != 255) & (img[k, j, 0] != 70):
#                 img[k, j, 0] = 0
#                 img[k, j, 1] = 0
#                 img[k, j, 2] = 0
#     cv.imshow("1", img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()


def partial_detect():
    """
    cv2.Canny(image,            # 输入原图（必须为单通道图）
              threshold1,
              threshold2,       # 较大的阈值2用于检测图像中明显的边缘
              [, edges[,
              apertureSize[,    # apertureSize：Sobel算子的大小
              L2gradient ]]])   # 参数(布尔值)：
                                  true： 使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开放），
                                  false：使用L1范数（直接将两个方向导数的绝对值相加）。
    """

    input_dir = "D://deeplabv3-plus-pytorch-main//img"
    output_dir = "D://deeplabv3-plus-pytorch-main//img_out"
    save_dir = "D://deeplabv3-plus-pytorch-main//img_out_new"

    output_list = os.listdir(output_dir)
    input_list = os.listdir(input_dir)
    for input_file, output_file in zip(input_list, output_list):
        print(input_file)
        input_img = cv2.imread(os.path.join(input_dir, input_file))
        output_img = cv2.imread(os.path.join(output_dir, output_file))

        # 将预测图像中的背景像素换成黑
        # h, w = output_img.shape[0], output_img.shape[1]
        # for i in range(h):
        #     for j in range(w):
        #         if (output_img[i, j, 0] != 200) | (output_img[i, j, 1] != 255) | (output_img[i, j, 2] != 70):
        #             output_img[i, j, 0] = 0
        #             output_img[i, j, 1] = 0
        #             output_img[i, j, 2] = 0

        # canny(): 边缘检测
        img1 = cv2.GaussianBlur(output_img, (3, 3), 0)
        canny = cv2.Canny(img1, 50, 150)

        # cv2.imshow("original_img", canny)

        # canny.shape = (h, w)
        # 获取预测图像的边框
        h, w = canny.shape[0], canny.shape[1]

        # 增加边缘宽度
        img_edge = magnify_edge(canny, h, w)
        # cv2.imshow("edge", img_edge)

        for i in range(h):
            for j in range(w):
                if img_edge[i, j] != 0:
                    input_img[i, j, 0] = 200
                    input_img[i, j, 1] = 255
                    input_img[i, j, 2] = 70

        # cv2.imshow("original_img", original_img)
        # cv2.imshow("paste_img", input_img)
        # cv2.imshow('Canny', canny)

        save_path = os.path.join(save_dir, input_file)
        cv2.imwrite(save_path, input_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def magnify_edge(img_edge, h, w):
    """增强边缘宽度，使边缘更加明显"""
    for i in range(h):
        for j in range(w):
            if img_edge[i, j] != 0:
                if (i > 0) & (j > 0):
                    img_edge[i - 1, j - 1] = img_edge[i, j]
                if j > 0:
                    img_edge[i, j - 1] = img_edge[i, j]
                if i > 0:
                    img_edge[i - 1, j] = img_edge[i, j]
    return img_edge

partial_detect()