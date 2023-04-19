import cv2 as cv
from PIL import Image
import os
import random
from torch.autograd import Variable
from torch import optim
from torch import nn


def PIL_mixup():
    img1 = Image.open("img/dust.jpg").convert("RGBA")
    img2 = Image.open("img/fire01_1.jpg").convert("RGBA").resize(img1.size)

    img = Image.blend(img1, img2, 0.8)
    img.show()


def model_mixup(loader1, loader2, model):
    optimizier = optim.SGD()
    loss_func = nn.CrossEntropyLoss()

    for (x1, y1), (x2, y2) in zip(loader1, loader2):
        lam = random.uniform(0, 1)
        x = Variable(lam * x1 + (1 - lam) * x2)
        y = Variable(lam * y1 + (1 - lam) * y2)
        loss = loss_func(model(x), y)

        optimizier.zero_grad()
        loss.backward()
        optimizier.step()
