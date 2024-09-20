from django.db import models
from datetime import datetime, timezone, timedelta
# Create your models here.
class UserInfo(models.Model):
    username = models.CharField(blank=False, max_length=150)
    sex = models.CharField(blank=False, max_length=10)
    chats = models.JSONField(blank=True, null=True)