from django.db import models
from datetime import datetime, timezone, timedelta
# Create your models here.
class UserInfo(models.Model):
    username = models.CharField(blank=False, max_length=150)
    sex = models.CharField(blank=True, max_length=10)
    job = models.CharField(blank=True, max_length=50)
    housing = models.CharField(blank=True, max_length=50)
    savings = models.CharField(blank=True, max_length=50)
    checking = models.CharField(blank=True, max_length=50)
    amount = models.CharField(blank=True, max_length=50)
    duration = models.CharField(blank=True, max_length=50)
    purpose = models.CharField(blank=True, max_length=50)
    chats = models.JSONField(blank=True, null=True)