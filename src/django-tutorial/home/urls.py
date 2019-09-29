# from django.conf.urls import url, include
from django.urls import path, include
from . import views

app_name = 'home'
urlpatterns = [
    # url(r'^$', views.video_list, name='list'),
    # url(r'^new$', views.video_new, name='new'),
    # url(r'^(?P<video_id>\d+)/$', views.video_detail, name='detail'),
    path('', views.home, name='home')
]