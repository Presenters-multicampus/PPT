from django.shortcuts import render, redirect, get_object_or_404
from .models import FileModel
from .forms import FileForm
from .snapshot_image import snapshot
from django.conf import settings
import os, shutil
from .tasks import snapshot_celery

# Create your views here.
SEC = 6  # 몇초마다 snapshot 할지 결정

def home(request):
    files = FileModel.objects.all()
    return render(request, 'home.html', {'files':files})

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

def video_detail(request, pk):
    video = get_object_or_404(FileModel, pk=pk)
    videoname = os.path.basename(video.file.url)  # 선택된 video name
    snaps_dir = os.path.join(settings.SNAPS_DIR, videoname)  # 선택된 video들의 snapshot 들이 저장되어 있는 디렉토리 경로
    snapshots = os.listdir(snaps_dir)  # 전체 snapshot 리스트
    snapshots = sorted(snapshots, key=lambda x: int(x.split('.')[0]))
    snapshots = list(map(lambda x: os.path.join('snapshots', videoname, x), snapshots))  # 숫자 기준으로 정렬

    # [[index, snapshot], [index, snapshot], ... ] 형태의 리스트로 변환
    snapshots = [[idx, snapshot] for idx, snapshot in enumerate(snapshots)]
    return render(request, 'video_detail.html', {'video':video, 'snapshots':snapshots})

# 클릭된 스냅샷 렌더링
def video_snapshot(request, pk, idx):
    idx = int(idx)
    # video_detail 과 동일
    video = get_object_or_404(FileModel, pk=pk)
    videoname = os.path.basename(video.file.url)
    snaps_dir = os.path.join(settings.SNAPS_DIR, videoname)
    snapshots = os.listdir(snaps_dir)
    snapshots = sorted(snapshots, key=lambda x: int(x.split('.')[0]))
    snapshots = list(map(lambda x: os.path.join('snapshots', videoname, x), snapshots))
    
    dict_snapshots = {index:snapshot for index, snapshot in enumerate(snapshots)}

    # [[index, snapshot], [index, snapshot], ... ] 형태의 리스트로 변환
    snapshots = [[idx, snapshot] for idx, snapshot in enumerate(snapshots)]

    snapshot = dict_snapshots[idx]  # 선택된 스냅샷
    sec = idx*SEC
    return render(request, 'video_detail.html', {'video':video, 'snapshots':snapshots, 'snap':snapshot, 'sec':sec})

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