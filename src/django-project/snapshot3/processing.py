import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from django.conf import settings


### 전역변수 선언
protoFile = os.path.join(settings.MODEL_DIR, 'caffemodel', 'pose_deploy_linevec.prototxt')
weightFile = os.path.join(settings.MODEL_DIR, 'caffemodel', 'pose_iter_440000.caffemodel')
network = cv2.dnn.readNetFromCaffe(protoFile, weightFile)

face_protoFile = os.path.join(settings.MODEL_DIR, 'caffemodel', 'deploy.prototxt')
face_weightFile = os.path.join(settings.MODEL_DIR, 'caffemodel', 'res10_300x300_ssd_iter_140000.caffemodel')

weight_path = os.path.join(settings.MODEL_DIR, 'emotion_detector_models', 'model_v6_23.hdf5')
weight_path2 = os.path.join(settings.MODEL_DIR, 'emotion_detector_models', 'model_resnet_best.hdf5')

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

# list => dict
def li2dict(points_list):
    tmp_li = []
    for k in range(len(points_list)):
        t = {points_list[k][i][2]: (points_list[k][i][0],points_list[k][i][1],points_list[k][i][3]) for i in range(len(points_list[k]))}
        tmp_li.append(t)
    return tmp_li

# 각 emotion의 등장 횟수 및 pos / neg의 percent 제공
def countEmotions(emotions):
    label_dict = {'Angry':'negative', 'Disgust':'negative', 'Fear':'negative', 'Happy':'positive', \
                  'Neutral':'neutral', 'Sad':'negative', 'Surprise':'positive', 'None':'neutral'}
    emotion_counts = {'positive' : 0, 'negative' : 0, 'neutral' : 0}
    for emotion in emotions:
        emotion_counts[label_dict[emotion]] += 1
        
    sum_emo_cnt = emotion_counts['positive'] + emotion_counts['negative'] + \
                    emotion_counts['neutral']
    
    per_pos = emotion_counts['positive'] / sum_emo_cnt
    per_neg = emotion_counts['negative'] / sum_emo_cnt
    per_neut = emotion_counts['neutral'] /sum_emo_cnt
    
    return emotion_counts, (per_pos, per_neg, per_neut)

# def snapshot_scoring(points, emotion):
#     score = 0

#     ## emotion 표정을 활용한 점수화 알고리즘
#     emotion_score = {'Angry': -1, 'Disgust': -1, 'Fear': -1, 'Happy': 1, 'Neutral': 1, 'Sad': -1, 'Surprise': 0, 'None':0}  # 임의 설정
#     score += emotion_score[emotion]  # 표정에 따른 점수

#     ## points_with_num 을 활용한 점수화 알고리즘
#     eyecount = countEyes(points)
#     score += 1 if eyecount==2 else -1

#     return score

### 눈 관련 함수
def countEyes(points):
    eyecount = 0
    if 14 in points.keys() :
        eyecount += 1
    if 15 in points.keys() :
        eyecount += 1
    return eyecount

def countTotalEyes(points_list):
    total_eye_cnt = 0
    see_front_cnt = 0
    len_snapshot = len(points_list)
    for points in points_list:
        pt = countEyes(points)
        total_eye_cnt += pt
        if pt:
            see_front_cnt += 1
    per_see_front = total_eye_cnt / (len_snapshot*2)
    return total_eye_cnt, see_front_cnt, per_see_front

def eyeAdvice(input_advice):
    if input_advice == 2:
        return '정면을 잘 응시하고 있네요!'
    elif input_advice == 1:
        return '측면을 응시하고 있네요!'
    else:
        return '이런! 발표중에 청중에게 등을 돌리는 것은 좋지 않아요!'

def eyeAdvice_group(input_advice):
    if input_advice[2] < 0.7:
        return '전체적으로 ppt 화면만 보고 이야기 하고 있어요!'
    elif input_advice[2] < 0.95:
        return '화면과 청중을 번갈아 보며 이야기 하고 있어요!'
    else:
        return '청중을 대부분 보며 이야기 하고 있어요!'

### hand moving 관련 함수
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import time

def checkHandMoving(points_list, save_path='./'):
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
    li = []
     # right shoulder, right elbow, rirhg wrist / left s, left e, left w
    for i in range(6):
        li.append([])
    for points in points_list:
        if 2 in points.keys():
            li[0].append(points[2][0:2])
        if 3 in points.keys():
            li[1].append(points[3][0:2])
        if 4 in points.keys():
            li[2].append(points[4][0:2])
        if 5 in points.keys():
            li[3].append(points[5][0:2])
        if 6 in points.keys():
            li[4].append(points[6][0:2])
        if 7 in points.keys():
            li[5].append(points[7][0:2])
    
#     print(li)
    
    mean_std_list = [] # order is same
    for i in range(6):
        p_list = np.array(li[i])
        p_std = p_list.std(axis=0).sum()
        p_mean = p_list.mean(axis=0).sum()
        mean_std_list.append([p_mean, p_std])
#         print(p_list)
#         print(p_std)
#         print(p_mean)
        
    df = pd.DataFrame(mean_std_list, columns=['p_mean','p_std'])
#     print(df)
        
    # matplotlib => savefig로 보여주는 것이 좋겠다
    # var, difference의 변화과정 보여주기
    
    f, ax = plt.subplots(1,1,figsize=(10,8))
#     f.set_title('Mean & Variance of arms move')
    df['p_std'].plot(kind='bar', ax=ax)
    filename = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    filename = filename + "_mean_std_plot"
    plt.savefig(os.path.join(save_path,filename))
    
    return mean_std_list

def adviceHandMoving_group(input_advice):
    rMove = (input_advice[1][1] + input_advice[2][1]) / input_advice[0][1]
    lMove = (input_advice[4][1] + input_advice[5][1]) / input_advice[3][1]
    if rMove + lMove < 10:
        return '몸이 너무 경직되어 있어요! 제스처를 좀 더 사용해 보는 건 어떨까요?'
    else:
        return '제스처를 적절히 사용하여 발표했어요!'

### standing straightly 함수
# 화면 1개의 점수
import math
import numpy as np

def checkStandingStraight(points):
    # check starightness between the middle point of LHip, RHip and Neck
    is_straight = False
    if 1 in points.keys() and 8 in points.keys() and 11 in points.keys():
        neck = np.array(points[1][0:2])
        rhip = np.array(points[8][0:2])
        lhip = np.array(points[11][0:2])
        mp = (rhip + lhip) / 2
#         ap = np.array([mp[0],neck[1]])
        val = math.sqrt(math.pow(mp[0]-neck[0],2) + math.pow(mp[1]-neck[1],2))
        cos = abs(mp[1]-neck[1]) / val
        rad = math.acos(cos)
        if rad * 57.296 < 5:
            is_straight = True
        return (True, rad * 57.296, is_straight) # exist, degree, straight or not
    else:
        return (False, 0, is_straight)

def standAdvice(input_advice):
    if input_advice[0]:
        if input_advice[2]:
            return (input_advice[1], '허리를 곧게 펴고 발표중이군요!')
        else:
            return (input_advice[1], '허리를 핀 상태로 발표하는 자세가 더 효과적입니다!')
    else:
        return ('측정 실패', '측정 실패')

import numpy as np
def checkStandingStraight_group(points_list):
    # check starightness between the middle point of LHip, RHip and Neck
    res_list = np.zeros(3) # cnt_ok, avg_degree, straight_percent
    for points in points_list:
        is_ok, degree, is_st = checkStandingStraight(points)
        if is_ok:
            res_list[0] += 1
            res_list[1] += degree
            if is_st:
                res_list[2] += is_st
    
    return res_list / res_list[0]

def adviceStandingStraight_group(input_advice):
    if input_advice[2] < 0.8:
        return (input_advice[1], '전체적으로 허리를 더 펴는 것이 좋습니다!')
    else:
        return (input_advice[1], '곧은 자세로 잘 발표하였습니다!')

### right-left balance 함수
def checkLeftRightBalance(points):
    is_balanced = False
    if 2 in points.keys() and 5 in points.keys() and 8 in points.keys() and 11 in points.keys():
        rsh = np.array(points[2][0:2])
        lsh = np.array(points[5][0:2])
        val = math.sqrt(math.pow(rsh[0]-lsh[0],2) + math.pow(rsh[1]-lsh[1],2))
        cos = abs(rsh[1]-lsh[1]) / val
        s_rad = math.acos(cos)
        
        rhip = np.array(points[8][0:2])
        lhip = np.array(points[11][0:2])
        val = math.sqrt(math.pow(rhip[0]-lhip[0],2) + math.pow(rhip[1]-lhip[1],2))
        cos = abs(rhip[1]-lhip[1]) / val
        h_rad = math.acos(cos)
        
        if 90 - ((s_rad + h_rad) * 57.296) / 2 < 7.5:
            is_balanced = True
        return (True, 90 - s_rad * 57.296, 90 - h_rad * 57.296, is_balanced) # exist, s_degree, h_degree, straight or not
    else:
        return (False, 0, 0, is_balanced)

def adviceLeftRightBalance_group(input_advice):
    if input_advice[3] < 0.8:
        return (input_advice[1], input_advice[2], '전체적으로 한 쪽으로 몸이 기울었습니다!')
    else:
        return (input_advice[1], input_advice[2], '한 쪽으로 치우치지 않은 자세로 잘 발표하였습니다!')

def balanceAdvice(input_advice):
    if input_advice[0]:
        if input_advice[3]:
            return (input_advice[1], input_advice[2], '양 쪽의 밸런스가 맞게 서있군요!')
        else:
            return (input_advice[1], input_advice[2], '몸이 한 쪽으로 기울어져 있군요!')
    else:
        return ('측정 실패', '측정 실패', '측정 실패')

# group
import numpy as np
def checkLeftRightBalance_group(points_list):
    # check starightness between the middle point of LHip, RHip and Neck
    res_list = np.zeros(4) # cnt_ok, avg_degree_s, avg_degree_h, balanced_percent
    for points in points_list:
        is_ok, degree_s, degree_h, is_bal = checkLeftRightBalance(points)
#         print(checkLeftRightBalance(points))
        if is_ok:
            res_list[0] += 1
            res_list[1] += degree_s
            res_list[2] += degree_h
            if is_bal:
                res_list[3] += is_bal
#         print(res_list)
    return res_list / res_list[0]

### 팔짱 낀 것은 아닌지
def checkCrossArms(points):
    is_crossed = False
    if 2 in points.keys() and 4 in points.keys() and 5 in points.keys() and 7 in points.keys():
        if (points[2][0] - points[5][0]) * (points[4][0] - points[7][0]) < 0:
            is_crossed = True
        return (True, is_crossed) # is_ok, is_crossed
    else:
        return (False, is_crossed)

def crossArmsAdvice(input_advice):
    if input_advice[0]:
        if input_advice[1]:
            return '발표시 팔짱은 좋지 않습니다!'
        else:
            return '팔을 잘 편 자세로 발표하고 있군요!'
    else:
        return '팔을 잘 편 자세로 발표하고 있군요!'
    

def checkCrossArms_group(points_list):
    res_list = np.zeros(2)
    for points in points_list:
        is_ok, is_crossed = checkCrossArms(points)
        if is_ok:
            res_list[0] += 1
            if is_crossed:
                res_list[1] += 1
    
    return res_list / res_list[0]

def adviceCrossArms_group(input_advice):
    if input_advice[1] > 0.2:
        return '팔짱을 무의식적으로 계속 끼고 있습니다!'
    else:
        return '바른 자세로 잘 발표하였습니다!'

### 머리 위에 손을 얹은 것은 아닌지

def checkPutHandsOnHead(points):
    is_put = False
    if 4 in points.keys() and 1 in points.keys() and 7 in points.keys():
        if points[1][1] > points[4][1] and points[1][1] > points[7][1]:
            is_put = True
        return (True, is_put)
    else:
        return (False, is_put)

def putHandOnHeadAdvice(input_advice):
    if input_advice[0]:
        if input_advice[1]:
            return '발표시 머리 위에 손을 얹는 자세는 좋지 않습니다!'
        else:
            return '좋은 자세로 발표하고 있군요!'
    else:
        return '좋은 자세로 발표하고 있군요!'

def checkPutHandsOnHead_group(points_list):
    res_list = np.zeros(2)
    for points in points_list:
        is_ok, is_put = checkPutHandsOnHead(points)
        if is_ok:
            res_list[0] += 1
            if is_put:
                res_list[1] += 1
    
    return res_list / res_list[0]  

# 최종 스코어 계산 함수
def total_scoring(points_list, emo_list):
    final_score = 0
    final_score += countEmotions(emo_list)[0] * 30
    final_score += countTotalEyes(points_list)[3] * 40
    if adviceHandMoving_group(checkHandMoving(points_list)) == '제스처를 적절히 사용하여 발표했어요!':
        final_score += 30
    else:
        final_score += 15
    final_score -= adviceStandingStraight_group(checkStandingStraight_group(points_list))[0]
    final_score -= adviceLeftRightBalance_group(checkLeftRightBalance_group(points_list))[0]
    final_score -= adviceLeftRightBalance_group(checkLeftRightBalance_group(points_list))[1]
    final_score -= checkCrossArms_group(points_list)[1]
    return final_score

# def countEyes(points):
#     eyecount = 0
#     for point in points:
#         if point[2] == 14 or point[2] == 15:
#             eyecount += 1
#     return eyecount

# emotion_detection 관련 함수

def resnet_layer(inputs,
                num_filters=16,
                kernel_size=3,
                strides=1,
                activation='relu',
                batch_normalization=True,
                conv_first=True):
    conv = Conv2D(num_filters,
                 kernel_size=strides,
                 padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=7):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
    input_shape (tensor): shape of input image tensor
    depth (int): number of core convolutional layers
    num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
    model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)
    
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs,
                    num_filters=num_filters_in,
                    conv_first=True)
    
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            # num of param setting 
            if stage == 0: # first stage
                num_filters_out = num_filters_in * 4
                if res_block == 0: # first layer & first stage
                    activation = None
                    batch_normalization = False
            else: # second, third stage
                num_filters_out = num_filters_in * 2
                if res_block == 0: # first layer but no first stage
                    strides = 2 # downsample
            y = resnet_layer(inputs=x,
                            num_filters=num_filters_in,
                            kernel_size=1,
                            strides=strides,
                            activation=activation,
                            batch_normalization=batch_normalization,
                            conv_first=False)
            y = resnet_layer(inputs=y,
                            num_filters=num_filters_in,
                            conv_first=False)
            y = resnet_layer(inputs=y,
                            num_filters=num_filters_out,
                            kernel_size=1,
                            conv_first=False)
            if res_block == 0:
                # Linear projection residual shortcut connection to match
                # changed dims
                # at the first time, make a shortcut origin
                x = resnet_layer(inputs=x,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                strides=strides,
                                activation=None,
                                batch_normalization=False)
            # and add every reputation
            x = keras.layers.add([x, y])
        
        num_filters_in = num_filters_out
    
    # Add classifier on top
    # v2 has BN_ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                   activation='softmax',
                   kernel_initializer='he_normal')(y)
    
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def load_model(weight_path=weight_path2):
    # Create the model
    n = 6
    version = 2
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    input_shape = [48,48,1]

    model = resnet_v2(input_shape=input_shape, depth=depth)
    
    model.load_weights(weight_path)
    
    return model

# def load_model():
#     # Create the model
#     model = Sequential()

#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,1)))
#     # model.add(BatchNormalization())

#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
#     # model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
#     # model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
#     # model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
#     # # model.add(BatchNormalization())

#     model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
#     # model.add(BatchNormalization())

#     model.add(Flatten())
#     model.add(Activation("softmax"))
#     model.load_weights(weight_path)
    
#     return model

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
