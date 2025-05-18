from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('image/', views.Radiologue_list, name='radiologue'),
    path('predict/', views.Predict, name='predict'),
    path('upload/', views.radiologue_upload, name='radiologue_upload'),
    path('download-report/', views.download_report_pdf, name='download_report'),
    path('report/', views.show_report, name='show_report'),
    path('logout/', LogoutView.as_view(next_page='index'), name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard')


]
#     path('list/Precoce.html', views.precoce_view, name='precoce'),



