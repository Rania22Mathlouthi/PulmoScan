from django.urls import path
from .views import user_login, user_logout, dashboard_redirect, profilePatient, profileRadiologue

urlpatterns = [
    path('login/', user_login, name='login'),  # Page de connexion
    path('logout/', user_logout, name='logout'),  # Page de déconnexion
    path('dashboard/', dashboard_redirect, name='dashboard'),  # Redirection vers le tableau de bord
    path('patient/', profilePatient, name='profilePatient'),  # Profil pour patient
    path('radiologue/', profileRadiologue, name='profileRadiologue'),  # Profil pour radiologue
]
