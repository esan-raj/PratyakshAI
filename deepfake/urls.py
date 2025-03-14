from django.contrib import admin
from django.urls import path
# from detection import views
from . import views

urlpatterns = [
    path("", views.index, name='home'),
    path("contact", views.contact, name='contact'),
    path("services", views.services, name='services'),
    path("real_video/", views.real_video, name='real_video'),
    path("upload_video/", views.upload_video, name='upload_video'),
    path("upload_photo/", views.upload_photo, name='upload_photo'),
    path("process_photo/", views.process_photo, name="process_photo"),
    path("process_video/", views.process_video, name="process_video"),
    path("about", views.about, name='about'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('process_input/', views.process_input, name='process_input'),
    path('deepfake_detection', views.detect_deepfake, name='detect_deepfake'),
    path('deepfake_video/', views.detect_deepfake_video, name='detect_deepfake_video'),
]
