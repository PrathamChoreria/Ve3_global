from django.urls import path
from . import views

urlpatterns = [
    path('', views.visuals, name='index'),
    path('delete-file/', views.delete_file, name='delete_file'),
    path('remove_outliers/',views.remove_outliers,name='remove_outliers')
]
