from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include


from django.views.generic.base import RedirectView
from rest_framework import routers
from hrpred import views
#from rest_framework_jwt.views import obtain_jwt_token

router = routers.DefaultRouter() #instantiate the DefaultRouter() object from routers (inbuilt Django Rest Framework)
router.register(r'hrpred', views.AttritionViewSet) #register titanic and use the url from TitanicViewSet
#router.register(r'auth', views.UserViewSet)

urlpatterns = [
    path(r'api/', include(router.urls)), #When there's an api, use the router
    path('',RedirectView.as_view(url='api/')), #when it's blank, go to 'api/'
    
    #url(r'api-token-auth/', obtain_jwt_token),
]
