from django.shortcuts import render, redirect, get_object_or_404
from .models import FileModel
from .forms import FileForm
from .snapshot_image import snapshot
from django.conf import settings
import os, shutil
from .tasks import snapshot_celery
import pickle
from .processing import *


# Create your views here.
SEC = 10  # 몇초마다 snapshot 할지 결정

# 홈 페이지 렌더링
def home(request):
    files = FileModel.objects.all()
    return render(request, 'home.html', {'files':files})

# 새 video 추가 및 'snapshot' 함수 실행
# split, object inferencing
def video_new(request):
    if request.method == 'POST':
        file = FileModel()
        form = FileForm(request.POST, request.FILES, instance=file)
        if form.is_valid():
            form.save()  # 파일 저장
            files = FileModel.objects.all()
            url = files[len(files)-1].file.url  # 아까 저장된 파일 url
            # save_dir = snapshot(url, SEC)  # 두번째 인자가 SEC, '몇초마다 snapshot 할지 결정'
            save_dir = snapshot_celery.delay(url, SEC)
            return redirect('home')
    else:
        form = FileForm()
    return render(request, 'video_new.html', {'form':form})

# 홈페이지에서 제목 클릭했을 시 비디오 첫화면 렌더링
def video_detail(request, pk):
    video = get_object_or_404(FileModel, pk=pk)
    videoname = os.path.basename(video.file.url)  # 선택된 video name
    snaps_dir = os.path.join(settings.SNAPS_DIR, videoname)  # 선택된 video들의 snapshot 들이 저장되어 있는 디렉토리 경로
    files = os.listdir(snaps_dir)  # 전체 snapshot 폴더의 파일 리스트
    snapshots = []
    for file in files:
        ext = file.split('.')[1]
        if ext=='jpg':
            snapshots.append(file)
    print(snapshots)
    snapshots = sorted(snapshots, key=lambda x: int(x.split('.')[0]))
    snapshots = list(map(lambda x: os.path.join('snapshots', videoname, x), snapshots))  # 숫자 기준으로 정렬

    # [[index, snapshot], [index, snapshot], ... ] 형태의 리스트로 변환
    snapshots = [[idx, snapshot] for idx, snapshot in enumerate(snapshots)]
    return render(request, 'video_detail.html', {'video':video, 'snapshots':snapshots})

# 클릭된 스냅샷 렌더링
def video_snapshot(request, pk, idx):
    # idx = int(idx)
    # video_detail 과 동일
    video = get_object_or_404(FileModel, pk=pk)
    videoname = os.path.basename(video.file.url)
    snaps_dir = os.path.join(settings.SNAPS_DIR, videoname)
    files = os.listdir(snaps_dir)  # 전체 snapshot 폴더의 파일 리스트
    snapshots = []
    for file in files:
        ext = file.split('.')[1]
        if ext=='jpg':
            snapshots.append(file)
    print(snapshots)
    snapshots = sorted(snapshots, key=lambda x: int(x.split('.')[0]))
    snapshots = list(map(lambda x: os.path.join('snapshots', videoname, x), snapshots))
    
    dict_snapshots = {index:snapshot for index, snapshot in enumerate(snapshots)}

    # [[index, snapshot], [index, snapshot], ... ] 형태의 리스트로 변환
    snapshots = [[idx, snapshot] for idx, snapshot in enumerate(snapshots)]

    # read emo.pkl, point.pkl, score.pkl file
    # snapshot 함수에서 emo.pkl, point.pkl, score.pkl 생성 후 여기서 불러옵니다.
    # 경로 변수 선언
    emo_path = os.path.join(snaps_dir, 'emo.pkl')
    point_path = os.path.join(snaps_dir, 'point.pkl')
    score_path = os.path.join(snaps_dir, 'score.pkl')
    # pickle 로 불러오기
    with open(emo_path, 'rb') as f:
        # emo_list[0] --> 0번째 스냅샷의 emotion
        emo_list = pickle.load(f)

    with open(point_path, 'rb') as f:
        # point_list[0] --> 0번째 스냅샷의 좌표 넘파이 배열 [[x, y, number, name], [x,y, number, name], ...]
        point_list = pickle.load(f)

    # list형식 => dict형식
    point_list = list(li2dict(point_list))

    with open(score_path, 'rb') as f:
        # score_list[0] --> 0번째 스냅샷의 score
        score_list = pickle.load(f)  

    # video detail 창 들어가서, 좌측 이미지 클릭하면
    # python manage.py runserver 실행한 프롬프트에서 확인 가능
    print('+'*50)
    print(emo_list)
    print(point_list)
    print(score_list)
    print('+'*50)

    # snap_info 설명
    # snap_info[0]: 클릭된 스냅샷 이미지  --> (html) snap_info.0  이렇게 접근(장고 템플릿 html에선 리스트 원소를 '.' 으로 참조함)
    # snap_info[1]: 클릭된 스냅샷 emotion  --> (html) snap_info.1
    # snap_info[2]: 클릭된 스냅샷 좌표 numpy 배열  --> (html) snap_info.2
    # snap_info[3]: 클릭된 스냅샷 score  --> (html) snap_info.3
    snap_info = [dict_snapshots[idx], emo_list[idx], point_list[idx], score_list[idx]]
    # snap_info[4]: 스냅샷의 눈에 따른 어드바이스 --> (html) snap_info.4
    current_points = point_list[idx]
    snap_info.append(eyeAdvice(countEyes(current_points)))
    # snap_info[5]: 스냅샷의 허리 각도 --> (html) snap_info.5
    if isinstance(standAdvice(checkStandingStraight(current_points))[0], float):
        snap_info.append('{0:.02f}'.format(float(standAdvice(checkStandingStraight(current_points))[0])))
    else:
        snap_info.append(standAdvice(checkStandingStraight(current_points))[0])

    # snap_info[6]: 스냅샷의 허리 어드바이스 --> (html) snap_info.6
    snap_info.append(standAdvice(checkStandingStraight(current_points))[1])
    # snap_info[7]: 스냅샷의 어깨가 삐딱한 각도 --> (html) snap_info.7
    if isinstance(standAdvice(checkLeftRightBalance(current_points))[0], float):
        snap_info.append('{0:.02f}'.format(float(balanceAdvice(checkLeftRightBalance(current_points))[0])))
    else:
        snap_info.append('관측 실패')
                
    snap_info.append(balanceAdvice(checkLeftRightBalance(current_points))[1])
    # snap_info[9]: 자세 기울어짐 어드바이스 --> (html) snap_info.9
    snap_info.append(balanceAdvice(checkLeftRightBalance(current_points))[2])
    # snap_info[10]: 팔 꼼 어드바이스 --> (html) snap_info.10
    snap_info.append(crossArmsAdvice(checkCrossArms(current_points)))
    # snap_info[11]: 머리위 손 어드바이스 --> (html) snap_info.11
    snap_info.append(putHandOnHeadAdvice(checkPutHandsOnHead(current_points)))    

    sec = idx*SEC
    return render(request, 'video_detail.html', {'video':video, 'snapshots':snapshots, 'snap_info':snap_info, 'sec':sec})

def video_score(request, pk):

    video = get_object_or_404(FileModel, pk=pk)
    videoname = os.path.basename(video.file.url)
    snaps_dir = os.path.join(settings.SNAPS_DIR, videoname)
    files = os.listdir(snaps_dir)  # 전체 snapshot 폴더의 파일 리스트
    snapshots = []
    for file in files:
        ext = file.split('.')[1]
        if ext=='jpg':
            snapshots.append(file)
    print(snapshots)
    snapshots = sorted(snapshots, key=lambda x: int(x.split('.')[0]))
    snapshots = list(map(lambda x: os.path.join('snapshots', videoname, x), snapshots))
    
    dict_snapshots = {index:snapshot for index, snapshot in enumerate(snapshots)}

    # [[index, snapshot], [index, snapshot], ... ] 형태의 리스트로 변환
    snapshots = [[idx, snapshot] for idx, snapshot in enumerate(snapshots)]

    # read emo.pkl, point.pkl, score.pkl file
    # snapshot 함수에서 emo.pkl, point.pkl, score.pkl 생성 후 여기서 불러옵니다.
    # 경로 변수 선언
    emo_path = os.path.join(snaps_dir, 'emo.pkl')
    point_path = os.path.join(snaps_dir, 'point.pkl')
    score_path = os.path.join(snaps_dir, 'score.pkl')
    # pickle 로 불러오기
    with open(emo_path, 'rb') as f:
        # emo_list[0] --> 0번째 스냅샷의 emotion
        emo_list = pickle.load(f)

    with open(point_path, 'rb') as f:
        # point_list[0] --> 0번째 스냅샷의 좌표 넘파이 배열 [[x, y, number, name], [x,y, number, name], ...]
        point_list = pickle.load(f)

    # list형식 => dict형식
    point_list = list(li2dict(point_list))

    with open(score_path, 'rb') as f:
        # score_list[0] --> 0번째 스냅샷의 score
        score_list = pickle.load(f)  

    # video detail 창 들어가서, Check Total Score 클릭하면
    # python manage.py runserver 실행한 프롬프트에서 확인 가능
    print('+'*50)
    print(emo_list)
    print(point_list)
    print(score_list)
    print('+'*50)

    ### 여기서 scoring 하자 ###

    final_advices = []
    # final_advices[0]: 긍정적인 표정 비율 --> (html) final_advices.0
    if isinstance(countEmotions(emo_list[1][0]), float):
        final_advices.append('{0:.2f}'.format(float(countEmotions(emo_list)[1][0]) * 100))
    else:
        final_advices.append(countEmotions(emo_list)[1][0])
    # final_advices[1]: 부정적인 표정 비율 --> (html) final_advices.1
    if isinstance(countEmotions(emo_list[1][1]), float):
        final_advices.append('{0:.2f}'.format(float(countEmotions(emo_list)[1][1]) * 100))
    else:
        final_advices.append(countEmotions(emo_list)[1][1])
    # final_advices[2]: 중립적인 표정 비율 --> (html) final_advices.2
    if isinstance(countEmotions(emo_list[1][1]), float):
        final_advices.append('{0:.2f}'.format(float(countEmotions(emo_list)[1][2]) * 100))
    else:
        final_advices.append(countEmotions(emo_list)[1][2])
    # final_advices[3]: eye 어드바이스 --> (html) final_advices.3
    final_advices.append(eyeAdvice_group(countTotalEyes(point_list)))
    # final_advices[4]: 팔 어드바이스 --> (html) final_advices.4
    final_advices.append(adviceHandMoving_group(checkHandMoving(point_list)))
    # final_advices[5]: 허리 자세 평균 각도 --> (html) final_advices.5
    final_advices.append('{0:.2f}'.format(float(adviceStandingStraight_group(checkStandingStraight_group(point_list))[0])))
    # final_advices[6]: 허리 자세 조언 --> (html) final_advices.6
    final_advices.append(adviceStandingStraight_group(checkStandingStraight_group(point_list))[1])
    # final_advices[7]: 어깨 자세 평균 각도 --> (html) final_advices.7
    final_advices.append('{0:.2f}'.format(float(adviceLeftRightBalance_group(checkLeftRightBalance_group(point_list))[0])))
    # final_advices[8]: 엉덩이 자세 평균 각도 --> (html) final_advices.8
    final_advices.append('{0:.2f}'.format(float(adviceLeftRightBalance_group(checkLeftRightBalance_group(point_list))[1])))
    # final_advices[9]: 기울어짐 자세 조언 --> (html) final_advices.9
    final_advices.append(adviceLeftRightBalance_group(checkLeftRightBalance_group(point_list))[2])
    # final_advices[10]: 팔짱낀 자세 --> (html) final_advices.10
    final_advices.append(adviceCrossArms_group(checkCrossArms_group(point_list)))


    final_score = int(total_scoring(point_list, emo_list))

    ##########################

    return render(request, 'video_score.html', {'video':video, 'snapshots':snapshots, 'score':final_score, 'advice':final_advices})

# 홈페이지의 'clear db' 링크 버튼 클릭시 실행
# db 내 FileModel objects
# media/ 내 video 파일
# snapshot3/static/snapshots/ 내 snapshot 파일들
# 모두 삭제
def clear(request):
    me = settings.MEDIA_ROOT
    sn = settings.SNAPS_DIR
    
    if os.path.exists(me):
        shutil.rmtree(me)
    if os.path.exists(sn):
        shutil.rmtree(sn)
    os.makedirs(me)
    os.makedirs(sn)
    FileModel.objects.all().delete()
    return redirect('home')
