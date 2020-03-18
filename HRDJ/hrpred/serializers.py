from rest_framework import serializers
from hrpred.models import Attrition
from django.contrib.auth.models import User

class AttritionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Attrition
        fields = "__all__"
        #fields = ('PassengerId',	'Survived'	,'Pclass',	'Name'	,'Sex',	'Age',	'SibSp',	'Parch',	'Ticket',	'Fare'	,'Cabin'	,'Embarked')

# class UserSerializer(serializers.HyperlinkedModelSerializer): #this is for the login user (if needed)
#     class Meta:
#         model = User
#         fields= ('username','password')