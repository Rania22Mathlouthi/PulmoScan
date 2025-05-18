from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm
from .forms import RegistrationForm
from django.contrib import messages
from django.contrib.auth.models import Group
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from .models import CustomUser  # Votre modèle Account avec les rôles patient et radiologue

# Helper functions to check the user role
def is_patient(user):
    return user.role == 'patient'

def is_radiologue(user):
    return user.role == 'radiologue'

@login_required
def dashboard_redirect(request):
    if is_patient(request.user):
        return redirect('profilePatient')  # Redirect to profilePatient page for Patient
    elif is_radiologue(request.user):
        return redirect('profileRadiologue')  # Redirect to profileRadiologue page for Radiologue
    else:
        return HttpResponseForbidden("You are not authorized to view this page")
# Profile views
@login_required
def profilePatient(request):
    return render(request, 'index_patient.html')

@login_required
def profileRadiologue(request):
    return render(request, 'index_radiologue.html')

# Login view
def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, "Logged in successfully!")
                return redirect('dashboard')  # Redirige vers le tableau de bord après la connexion réussie
            else:
                messages.error(request, "Invalid credentials")
        else:
            messages.error(request, "Invalid form submission")
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

# Logout view
def user_logout(request):
    logout(request)
    messages.success(request, "Logged out successfully!")
    return redirect('index')
