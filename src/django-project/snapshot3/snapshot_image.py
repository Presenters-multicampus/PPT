import cv2
import numpy as np
import os, shutil
from django.conf import settings
from .processing import *
import pickle

def snapshot(url, SEC=3):
    filename = os.path.basename(url)
    url = os.path.join(settings.MEDIA_ROOT, filename)  # video 파일 가져올 경로
    cap = cv2.VideoCapture(url)
    save_dir = os.path.join(settings.SNAPS_DIR, filename)  # snapshots 저장할 경로

    if os.path.exists(save_dir):  # if filename is duplicated, remove that directory
        shutil.rmtree(save_dir)

    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print('Error: Creating directory of data')

    fps = round(cap.get(cv2.CAP_PROP_FPS))  # video 의 fps 값 얻기
    ret, frame = cap.read()
    
    

    cf = 0  # Current Frame
    emo_list = []  # emotion 들 저장하는 list
    point_list = []  # 각 snapshot 의 포인트 배열 저장하는 list
    score_list = []  # 각 snapshot 의 점수 저장하는 list
    # video -> snapshot 프레임 한장 한장 순회하는 반복문
    while(ret):
        # capture frame-by-frame
        ret, frame = cap.read()

        # saves image of the current frame in jpg file
        if cf%(fps*SEC) == 0:
            ## getSkeleton funcion call in 'preprocessing.py'
            img_skeleton, img_with_dot, points_with_num, emotion = getSkeleton(frame)

            current_score = 0  # scoring function call

            name = os.path.join(save_dir, f'{str(cf)}.jpg')  # ex) '100.jpg'
            print(f'Creating... {name}')
            cv2.imwrite(name, img_skeleton)

            emo_list.append(emotion)
            point_list.append(points_with_num)
            score_list.append(current_score)

        cf += 1

    cap.release()
    emo_path = os.path.join(save_dir, 'emo.pkl')
    point_path = os.path.join(save_dir, 'point.pkl')
    score_path = os.path.join(save_dir, 'score.pkl')


    # emotions, points, scores 리스트들 pickle 로 저장  --> 모든 snapshots 들에 대한 정보들입니다.
    # 경로: snapshot3/static/snapshots/videoname.mp4/emo.pkl 등
    with open(emo_path, 'wb') as f:
        pickle.dump(emo_list, f, pickle.HIGHEST_PROTOCOL)

    with open(point_path, 'wb') as f:
        pickle.dump(point_list, f, pickle.HIGHEST_PROTOCOL)

    with open(score_path, 'wb') as f:
        pickle.dump(score_list, f, pickle.HIGHEST_PROTOCOL)

    print('\n  Done. \n')
    return save_dir
