from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('list/', views.Patient_list, name="Patient_list"),
    path('list/Precoce.html', views.precoce_view, name='precoce'),
    path('logout/', LogoutView.as_view(next_page='index'), name='logout'),



   
]
