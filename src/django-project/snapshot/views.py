from django.shortcuts import render, redirect
from .models import File
# Create your views here.
def base_page(request):
    files = File.objects.all()
    return render(request, 'home.html', {'files':files})

def file_new(request):
    return redirect('file_new.html', {})