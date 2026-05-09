from django.db import models


class PredictionRecord(models.Model):
    audio_file   = models.FileField(upload_to='audio/')
    spectrogram  = models.ImageField(upload_to='spectrograms/', blank=True)
    emotion      = models.CharField(max_length=20, blank=True)
    confidence   = models.FloatField(null=True, blank=True)
    all_probs    = models.JSONField(null=True, blank=True)
    created_at   = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        conf = f'{self.confidence:.1%}' if self.confidence is not None else '?'
        ts   = self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else 'unsaved'
        return f"{self.emotion} ({conf}) — {ts}"

    def confidence_pct(self):
        return round(self.confidence * 100, 1) if self.confidence is not None else None
