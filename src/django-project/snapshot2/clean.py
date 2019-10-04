import os
import shutil
from django.conf import settings

def clean_dirs():
    me = settings.MEDIA_DIR
    sn = settings.SNAPS_DIR
    shutil.rmtree(me)
    # shutil.rmtree(sn)
    os.makedirs(me)
    # os.makedirs(sn)
