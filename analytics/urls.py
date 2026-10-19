from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('download-sample-csv/', views.download_sample_csv, name='download_sample_csv'),
    path('export-batch-csv/', views.export_batch_csv, name='export_batch_csv'),
    path('clear-history/', views.clear_history, name='clear_history'),
]
