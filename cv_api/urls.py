from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('', admin.site.urls),
    path('detect/', include('face_detector.urls')),
    
    

]
