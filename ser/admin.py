from django.contrib import admin
from .models import PredictionRecord


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ('emotion', 'confidence', 'created_at')
    list_filter = ('emotion',)
    ordering = ('-created_at',)
