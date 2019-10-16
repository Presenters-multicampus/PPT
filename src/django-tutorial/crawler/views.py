from django.shortcuts import render
from .models import RepositData
# Create your views here.

def reposit_list(request):
    reposits = RepositData.objects.all()
    return render(request, 'reposit/reposit_list.html', {'reposits': reposits})