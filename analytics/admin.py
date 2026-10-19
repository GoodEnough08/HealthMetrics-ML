from django.contrib import admin
from .models import PatientRecord

@admin.register(PatientRecord)
class PatientRecordAdmin(admin.ModelAdmin):
    list_display = ('name', 'glucose', 'bmi', 'age', 'bp', 'insulin', 'risk_percentage', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('name',)
    ordering = ('-created_at',)

