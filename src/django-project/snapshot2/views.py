from django.shortcuts import render, redirect
from .models import FileModel
from .forms import FileForm

# Create your views here.
def home(request):
    files = FileModel.objects.all()
    return render(request, 'home.html', {'files':files})

def file_new(request):
    files = FileModel.objects.all()
    if request.method == 'POST':
        form = FileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = FileForm()
    return render(request, 'file_new.html', {'form':form})