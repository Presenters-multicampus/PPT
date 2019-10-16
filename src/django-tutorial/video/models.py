from django.db import models

# Create your models here.

class Video(models.Model):
    title = models.CharField(max_length=200)
    video_key = models.CharField(max_length=12)