import os
from celery import Celery
from django.conf import settings

# set the default Django settings module for the 'celery' program
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

BROKER_URL = 'redis://127.0.0.1:6379/0'
CELERY_RESULT_BACKEND_URL = 'redis://127.0.0.1:6379/0'
app = Celery('snapshot3', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND_URL)

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
# should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks(settings.INSTALLED_APPS)
