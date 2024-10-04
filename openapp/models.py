from django.db import models
from django.contrib.auth import get_user_model
# Create your models here.
User = get_user_model()

class ChatGptBot(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    messageInput = models.TextField()
    bot_response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.user.username
    
    class Meta:
        verbose_name = 'Messages History'
        verbose_name_plural = 'Messages History'
        ordering = ['created_at']

# class ChatAnalytics(models.Model):
#     query_type = models.CharField(max_length=50)  # market_insight, price, etc.
#     response_time = models.FloatField()
#     user_satisfaction = models.IntegerField(null=True)
#     created_at = models.DateTimeField(auto_now_add=True)
    
#     class Meta:
#         verbose_name = 'Chat Analytics'