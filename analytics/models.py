from django.db import models

class PatientRecord(models.Model):
    name = models.CharField(max_length=100)
    glucose = models.FloatField()
    bmi = models.FloatField()
    age = models.IntegerField()
    bp = models.FloatField()
    insulin = models.FloatField()
    risk_percentage = models.FloatField()
    status = models.CharField(max_length=25)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Patient Record'
        verbose_name_plural = 'Patient Records'

    def __str__(self):
        return f"{self.name} - {self.risk_percentage}% ({self.status})"

    @property
    def badge_class(self):
        if self.risk_percentage >= 50.0:
            return 'danger'
        elif self.risk_percentage >= 30.0:
            return 'warning'
        return 'success'
