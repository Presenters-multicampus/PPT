from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('clear/', views.clear, name='clear'),
    path('video/new/', views.video_new, name='video_new'),
    path('video/<int:pk>', views.video_detail, name='video_detail')
]