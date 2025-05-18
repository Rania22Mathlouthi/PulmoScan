from django.shortcuts import render

# Vue pour afficher index.html
def index(request):
    return render(request, 'index.html')
