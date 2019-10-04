from django.shortcuts import render, redirect
from .models import FileModel
from .forms import FileForm
from .snapshot_image import snapshot
from .clean import clean_dirs
from django.conf import settings
import os

# Create your views here.
def home(request):
    files = FileModel.objects.all()
    if len(files) == 0:
        return render(request, 'home.html', {})

    # data
    data = []
    for file in files:
        if file.file.url[-3:] == 'mp4':
            data.append([file, True])
        else:
            data.append([file, False])

    # snapshots
    snapshots_dir = settings.SNAPS_DIR
    videos = os.listdir(snapshots_dir)
    return render(request, 'home.html', {'videos':videos})

def file_new(request):
    clean_dirs()  # media, snapshots directory reset
    if request.method == 'POST':
        form = FileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            files = FileModel.objects.all()
            url = files[len(files)-1].file.url
            save_dir = snapshot(url, 20)
            return redirect('home')
    else:
        form = FileForm()
    return render(request, 'file_new.html', {'form':form})

def video_detail(request, video_title):
    snaps_dir = os.path.join(settings.SNAPS_DIR, video_title)
    snapshots = os.listdir(snaps_dir)
    snapshots = sorted(snapshots, key=lambda x: int(x.split('.')[0]))
    snapshots = list(map(lambda x: os.path.join('snapshots', video_title, x), snapshots))
    
    files = FileModel.objects.all()
    if len(files) > 0:
        file = files[len(files)-1]
        print('hello', file.file.url)
        return render(request, 'video_detail.html', {'snapshots':snapshots, 'data':file})

    return render(request, 'video_detail.html', {'snapshots':snapshots})