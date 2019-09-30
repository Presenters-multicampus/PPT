from django.shortcuts import render
from .models import uploadedImage

# Create your views here.
def image_list(request):
    imgs = uploadedImage.objects.all()
    return render(request, 'image_list.html', {'imgs':imgs})