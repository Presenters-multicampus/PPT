from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

import os

# Create your models here.
class File(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    file = models.FileField()

    def __str__(self):
        return self.name

    def delete(self, *args, **kargs):
        os.remove(os.path.join(settings.MEDIA_ROOT, self.file.path))
        super(File, self).delete(*args, **kargs)
    