from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('file/new/', views.file_new, name='file_new')
]