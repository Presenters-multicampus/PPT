import cv2
import numpy as np
import os, shutil
from django.conf import settings
from .coco import getSkeleton

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
    while(ret):
        # capture frame-by-frame
        ret, frame = cap.read()

        # saves image of the current frame in jpg file
        if cf%(fps*SEC) == 0:
            frame, _, _ = getSkeleton(frame)
            name = os.path.join(save_dir, f'{str(cf)}.jpg')
            print(f'Creating... {name}')
            cv2.imwrite(name, frame)

        cf += 1

    cap.release()
    print('\n  Done. \n')
    return save_dir