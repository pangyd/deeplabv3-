import os
import time
import random
import cv2 as cv
from winsound import PlaySound
from subprocess import Popen
from playsound import playsound
from pygame import mixer


PlaySound("a.wav", flags=1)
Popen("a.mp3", shell=True)
playsound("a.mp3")
os.system("a.mp3")


def broadcast(file_path):
    """播放语音"""
    mixer.init()
    mixer.music.load(file_path)
    mixer.music.play()
    time.sleep(30)
    mixer.music.stop()


# 播放自我介绍问题
introduction = "自我介绍.mp3"
broadcast(introduction)

# 获取问题语音文件列表
path = ""
questions = os.listdir(path)
q_index = list(range(len(questions)))

# 停顿两分半
time.sleep(150)

for i in range(len(questions)):
    # 随机选取一个问题语音
    ind = random.choice(q_index)

    # 播放语音
    question = os.path.join(path, questions[ind])   # 问题音频路径
    broadcast(question)

    # 等待30s
    time.sleep(30)

    # 删除已提问的问题
    q_index.remove(ind)

    print("第{}个问题播放完毕".format(i+1))