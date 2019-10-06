from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
# Create your models here.

class FileModel(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField()

    def __str__(self):
        return self.title

    def delete(self, *args, **kargs):
        os.remove(os.path.join(settings.MEDIA_ROOT, self.file.path))
        super(FileModel, self).delete(*args, **kargs)