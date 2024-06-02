from django.urls import path
from . import views

urlpatterns = [
    path('', views.form_view, name='form'),
    path('predict/', views.predict, name='predict'),
]

