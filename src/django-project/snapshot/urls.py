from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.base_page, name='base_page'),
    path('snapshot/file_new/', views.file_new, name='file_new'),
]