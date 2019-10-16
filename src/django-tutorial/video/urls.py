from django.conf.urls import url, include
from django.urls import path, include
from . import views

app_name = 'video'
urlpatterns = [
    # url(r'^$', views.video_list, name='list'),
    # url(r'^new$', views.video_new, name='new'),
    url(r'^(?P<video_id>\d+)/$', views.video_detail, name='detail'),

    path('', views.video_list, name='video_list'),
    path('new/', views.video_new, name='video_new'),
    path('<int:pk>/', views.video_detail, name='video_detail'),
]