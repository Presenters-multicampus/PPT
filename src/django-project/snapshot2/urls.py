from django.urls import path, include
from . import views
from django.conf import settings
import os


# videos = os.listdir(settings.SNAPS_DIR)
# video_urls = []
# for video in videos:
#     video_urls.append(path(f'video/<{video}>/', views.video_detail, name='video_detail'))

urlpatterns = [
    path('', views.home, name='home'),
    path('file/new/', views.file_new, name='file_new'),
    path('video/<video_title>/', views.video_detail, name='video_detail'),
]