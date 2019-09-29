from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.image_list, name='image_list'),
]
