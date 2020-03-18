from django.db import models

class Attrition(models.Model):
    satisfaction_level = models.FloatField(max_length=50)
    last_evaluation = models.FloatField(max_length=50)
    number_project = models.IntegerField()
    average_montly_hours = models.IntegerField()
    time_spend_company = models.IntegerField()
    Work_accident = models.IntegerField()
    left = models.IntegerField(null=True, blank=True)
    promotion_last_5years = models.IntegerField()
    sales = models.CharField(max_length =10, null=True, blank =True)
    salary = models.CharField(max_length =10, null=True, blank =True)

#the data is converteed into a dictionary, since the object needs to fed into a dataframe. Dictionary converts into the dataframe.

    def to_dict(self): #dictionary helps in making a dataframe.
        return {
        'satisfaction_level': self.satisfaction_level,
        'last_evaluation': self.last_evaluation,
        'number_project': self.number_project,
        'average_montly_hours':self.average_montly_hours,
        'time_spend_company':self.time_spend_company,
        'Work_accident':self.Work_accident,
        'left':self.left,
        'promotion_last_5years':self.promotion_last_5years,
        'sales':self.sales,
        'salary':self.salary
        }

