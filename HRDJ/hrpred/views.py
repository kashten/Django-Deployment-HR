from django.http.response import HttpResponse
from rest_framework import viewsets
from hrpred.models import Attrition
from rest_framework.response import Response
from hrpred import HRML
from hrpred.serializers import AttritionSerializer
import os

#Change queryset and serializer_class.

class AttritionViewSet(viewsets.ModelViewSet): #ModelViewSet is a set of 5 views Django makes (create/delete/update etc).
    queryset = Attrition.objects.all()#.order_by('-id') #-id is descending order. Note: Titanic is the model here. Can also filter using Titanic.objects.filter(subject__icontains='exam')
    serializer_class = AttritionSerializer                #subject is the column and icontains is ignore case. Will search for 'exam' in the column regardless of case.

#this overwrites the method. View creates 5 sets and during the create phase, we're overwriting it since it not only has to store to DB but also show prediction.
    def create (self, request, *args, **kwargs):  
        viewsets.ModelViewSet.create(self,request,*args,**kwargs) #write on the db as usual.
        ob=Attrition.objects.latest('id') #pull the latest object and predict survival.
        attr=HRML.pred(ob) #Predict 
        return Response ({"Status": "Result Computed!", "1-Quit, 0-Stays": attr}) #returns json of prediction.

