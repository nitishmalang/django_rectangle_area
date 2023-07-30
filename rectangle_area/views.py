# rectangle_area/views.py

from django.shortcuts import render
from .models import Rectangle

def calculate_area(request):
    if request.method == 'POST':
        length = float(request.POST.get('length'))
        width = float(request.POST.get('width'))
        area = length * width
        return render(request, 'rectangle_area/area_result.html', {'area': area})
    return render(request, 'rectangle_area/area_form.html')
