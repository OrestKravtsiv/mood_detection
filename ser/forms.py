from django import forms
from django.conf import settings


class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(
        label='Audio file',
        help_text='Upload a .wav file (max 10 MB)',
    )

    def clean_audio_file(self):
        f = self.cleaned_data['audio_file']
        if not f.name.lower().endswith('.wav'):
            raise forms.ValidationError('Only .wav files are accepted.')
        max_bytes = getattr(settings, 'MAX_UPLOAD_SIZE_MB', 10) * 1024 * 1024
        if f.size > max_bytes:
            raise forms.ValidationError(f'File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB.')
        return f
