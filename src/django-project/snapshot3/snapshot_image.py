import cv2
import numpy as np
import os, shutil
from django.conf import settings
from .processing import *

def snapshot(url, SEC=3):
    filename = os.path.basename(url)
    url = os.path.join(settings.MEDIA_ROOT, filename)
    cap = cv2.VideoCapture(url)
    save_dir = os.path.join(settings.SNAPS_DIR, filename)

    if os.path.exists(save_dir):  # if filename is duplicated, remove that directory
        shutil.rmtree(save_dir)

    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print('Error: Creating directory of data')

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    cf = 0  # Current Frame
    emo_list = []
    while(ret):
        # capture frame-by-frame
        ret, frame = cap.read()

        # saves image of the current frame in jpg file
        if cf%(fps*SEC) == 0:
            img_skeleton, img_with_dot, points_with_num, emotion = getSkeleton(frame)
            name = os.path.join(save_dir, f'{str(cf)}.jpg')
            print(f'Creating... {name}')
            cv2.imwrite(name, img_skeleton)
            emo_list.append(emotion)

        cf += 1

    cap.release()
    emo_path = os.path.join(save_dir, 'emotions.txt')
    print(emo_list)
    with open(emo_path, 'w') as f:
        for i in emo_list:
            f.write(i+'\n')

    print('\n  Done. \n')
    return save_dir