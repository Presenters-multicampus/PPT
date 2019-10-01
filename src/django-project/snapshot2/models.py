from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class FileModel(models.Model):
    # owner = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    file = models.FileField(upload_to='media')

    def __str__(self):
        return self.title

        
