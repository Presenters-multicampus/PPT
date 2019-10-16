from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.reposit_list, name='reposit_list'),
]