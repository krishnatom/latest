from django.urls import path
from django.conf.urls import  url

from . import views



urlpatterns = [
path('', views.detect,name='detect'),

]
print(urlpatterns)
for x in range(10):
	print()