from django.urls import path
from . import views

app_name = 'ser'

urlpatterns = [
    path('',              views.hero_view,   name='hero'),
    path('upload/',       views.upload_view, name='upload'),
    path('api/predict/',  views.predict_api, name='predict_api'),
    path('result/<int:pk>/', views.result_view, name='result'),
    path('history/',      views.history_view, name='history'),
]
