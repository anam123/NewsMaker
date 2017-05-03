# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django import forms
from django.db import models
from django.forms import ModelForm, Textarea

class Question(models.Model):
    news_Article_Heading = models.CharField(max_length=2000)
    content = models.TextField()
    created = models.DateField(auto_now_add=True)
    def __str__(self):
        return self.news_Article_Heading



class Choice(models.Model):
    news_Article_Heading = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=2000)
    def __str__(self):
        return self.choice_text

class PostModelForm(ModelForm):
    class Meta:
        model = Question
        fields = '__all__'
        widgets = {
            'content': Textarea(attrs={'cols': 80, 'rows': 20}),
        }
