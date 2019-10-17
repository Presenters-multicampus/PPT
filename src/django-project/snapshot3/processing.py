import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from keras.regularizers import l1

import cv2
import os
import numpy as np
from django.conf import settings


### 전역변수 선언
protoFile = os.path.join(settings.MODEL_DIR, 'caffemodel', 'pose_deploy_linevec.prototxt')
weightFile = os.path.join(settings.MODEL_DIR, 'caffemodel', 'pose_iter_440000.caffemodel')
network = cv2.dnn.readNetFromCaffe(protoFile, weightFile)

face_protoFile = os.path.join(settings.MODEL_DIR, 'caffemodel', 'deploy.prototxt')
face_weightFile = os.path.join(settings.MODEL_DIR, 'caffemodel', 'res10_300x300_ssd_iter_140000.caffemodel')

weight_path = os.path.join(settings.MODEL_DIR, 'emotion_detector_models', 'model_v6_23.hdf5')
weight_path2 = os.path.join(settings.MODEL_DIR, 'emotion_detector_models', 'model_resnet_best_r2.hdf5')

face_cascade = cv2.CascadeClassifier(os.path.join(settings.MODEL_DIR, 'harrs', 'haarcascade_frontalface_default.xml'))


### 함수 선언
# 각 snapshot 한장에 대한 scoring
def personDetector(frame, pf=face_protoFile, wf=face_weightFile):
    global startX, endX, startY, endY
    net = cv2.dnn.readNetFromCaffe(pf,wf)
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return frame, (startX, startY, endX-startX, endY-startY)

def snapshot_scoring(points, emotion):
    score = 0

    ## emotion 표정을 활용한 점수화 알고리즘
    emotion_score = {'Angry': -1, 'Disgust': -1, 'Fear': -1, 'Happy': 1, 'Neutral': 1, 'Sad': -1, 'Surprise': 0, 'None':0}  # 임의 설정
    score += emotion_score[emotion]  # 표정에 따른 점수

    ## points_with_num 을 활용한 점수화 알고리즘
    eyecount = countEyes(points)
    score += 1 if eyecount==2 else -1

    return score

# 최종 스코어 계산 함수
def total_scoring(points, emotion, scores):
    final_score = 100
    final_advice = '연습하세요'
    return final_score, final_advice


def countEyes(points):
    eyecount = 0
    for point in points:
        if point[2] == 14 or point[2] == 15:
            eyecount += 1
    return eyecount

def load_model():
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,1)))
    # model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    # # model.add(BatchNormalization())

    model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Activation("softmax"))
    model.load_weights(weight_path)
    
    return model

# 'snapshot_image.py' 의 snapshot 함수 내에서 호출됨
# frame 은 'SEC'초 마다 나오는 이미지 1장
def getSkeleton(frame, isHarr=False):
    ### Inferencing Pose ###
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],\
                  [6,7],[1,8],[8,9],[9,10], [14,11], [11,12], [12,13] ]
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
    idx_2_BODY_PARTS = {int(num):part for part, num in BODY_PARTS.items()}
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1
    
    # inHeight / inWidth
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight),\
                                   (0,0,0), swapRB=False, crop=False)
    
    dotsize = int((frameWidth + frameHeight) / (inWidth))
    textsize = int((frameWidth + frameHeight) / (inWidth * 4))
    linesize = dotsize // 2
    
    network.setInput(inpBlob)
    
    output = network.forward()
    H = output.shape[2]
    W = output.shape[3]
    
    points = []
    points_with_num = []
    facelist = []
    
    ### Inferencing face & emotion ###
    if isHarr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facelist = face_cascade.detectMultiScale(gray, 1.3, 5)
        facelist = list(np.array(facelist).flatten())[:4]
    else:
        facelist = personDetector(frame)[1]
        facelist = list(np.array(facelist).flatten())
    
    emotion = 'None'  # 초기값 None 이 아니라 string 이어야 합니다.
    if len(facelist)!=0:
        (fx,fy,fw,fh) = list(map(int,facelist))
        # print(type(fx),type(fy),type(fw),type(fh))
        face = cv2.cvtColor(frame[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, dsize=(48, 48), interpolation=cv2.INTER_LINEAR)
        input_face= face.reshape(-1,48,48,1)
        model = load_model()
        label_dict = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
        label_dict = {num:emotion for emotion, num in label_dict.items()}
        emotion = label_dict[np.argmax(model.predict(input_face))]
        
        ## memory flush ##


    ### Draw points, lines, rectangle ###
    ### Pose estimation ###
    for i in range(nPoints):
        # confidence map of corresponding body's part
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        
        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), dotsize, (0,255,255),\
                      thickness=dotsize, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)),\
                       cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,255), textsize, \
                       lineType=cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), dotsize, (0, 0, 255),\
                       thickness=dotsize, lineType=cv2.FILLED)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
            points_with_num.append((int(x), int(y), int(i), idx_2_BODY_PARTS[int(i)]))
        else:
            points.append(None)
    
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0,255,255), linesize)
    
    ### Draw Face box and emotion ###
    #if len(facelist)!=0:
     #   cv2.rectangle(frame, (int(fx),int(fy)),(int(fx+fw),int(fy+fh)),(0,255,0), linesize)
      #  cv2.rectangle(frameCopy, (int(fx),int(fy)),(int(fx+fw),int(fy+fh)),(0,255,0), linesize)
    
    img_skeleton = frame
    img_with_dot = frameCopy
    return img_skeleton, img_with_dot, points_with_num, emotion
