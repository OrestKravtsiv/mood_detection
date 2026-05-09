import io
import json
import struct
import wave
from unittest.mock import patch, MagicMock

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, Client
from django.urls import reverse

from .forms import AudioUploadForm
from .models import PredictionRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_sec=1.0, sample_rate=22050) -> bytes:
    """Build a minimal valid WAV file in memory (silence)."""
    n_samples = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{n_samples}h', *([0] * n_samples)))
    return buf.getvalue()


def _wav_file(name='test.wav', duration_sec=1.0):
    return SimpleUploadedFile(name, _make_wav_bytes(duration_sec), content_type='audio/wav')


def _fake_prediction():
    return {
        'emotion': 'happy',
        'confidence': 0.92,
        'all_probs': {
            'happy': 0.92, 'sad': 0.03, 'angry': 0.02,
            'fearful': 0.01, 'neutral': 0.01, 'disgust': 0.005, 'surprised': 0.005,
        },
        'mel_spec_db': np.zeros((128, 130)),
    }


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class PredictionRecordModelTest(TestCase):

    def test_confidence_pct_returns_rounded_percentage(self):
        record = PredictionRecord(emotion='happy', confidence=0.923)
        self.assertEqual(record.confidence_pct(), 92.3)

    def test_confidence_pct_none_when_no_confidence(self):
        record = PredictionRecord(emotion='happy', confidence=None)
        self.assertIsNone(record.confidence_pct())

    def test_str_representation(self):
        record = PredictionRecord(emotion='sad', confidence=0.75)
        # Just confirm it doesn't crash and contains key info
        self.assertIn('sad', str(record))


# ---------------------------------------------------------------------------
# Form tests
# ---------------------------------------------------------------------------

class AudioUploadFormTest(TestCase):

    def test_valid_wav_accepted(self):
        form = AudioUploadForm(files={'audio_file': _wav_file()})
        self.assertTrue(form.is_valid(), form.errors)

    def test_non_wav_extension_rejected(self):
        mp3 = SimpleUploadedFile('clip.mp3', b'fake-mp3-data', content_type='audio/mpeg')
        form = AudioUploadForm(files={'audio_file': mp3})
        self.assertFalse(form.is_valid())
        self.assertIn('audio_file', form.errors)

    def test_oversized_file_rejected(self):
        big = SimpleUploadedFile('big.wav', b'x' * (11 * 1024 * 1024), content_type='audio/wav')
        form = AudioUploadForm(files={'audio_file': big})
        self.assertFalse(form.is_valid())
        self.assertIn('audio_file', form.errors)

    def test_missing_file_rejected(self):
        form = AudioUploadForm(files={})
        self.assertFalse(form.is_valid())


# ---------------------------------------------------------------------------
# View tests
# ---------------------------------------------------------------------------

class UploadViewTest(TestCase):

    def setUp(self):
        self.client = Client()
        self.url = reverse('ser:upload')

    def test_get_returns_200(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'ser/upload.html')

    def test_post_invalid_extension_stays_on_upload_page(self):
        mp3 = SimpleUploadedFile('clip.mp3', b'fake', content_type='audio/mpeg')
        response = self.client.post(self.url, {'audio_file': mp3})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'ser/upload.html')

    @patch('ser.views.ml_inference.generate_spectrogram_image')
    @patch('ser.views.ml_inference.predict_emotion', return_value=_fake_prediction())
    def test_post_valid_wav_redirects_to_result(self, mock_predict, mock_spec):
        mock_spec.return_value = None
        response = self.client.post(self.url, {'audio_file': _wav_file()})
        self.assertEqual(response.status_code, 302)
        self.assertIn('/result/', response['Location'])

    @patch('ser.views.ml_inference.generate_spectrogram_image')
    @patch('ser.views.ml_inference.predict_emotion', return_value=_fake_prediction())
    def test_post_creates_prediction_record(self, mock_predict, mock_spec):
        mock_spec.return_value = None
        self.client.post(self.url, {'audio_file': _wav_file()})
        self.assertEqual(PredictionRecord.objects.count(), 1)
        record = PredictionRecord.objects.first()
        self.assertEqual(record.emotion, 'happy')
        self.assertAlmostEqual(record.confidence, 0.92)

    @patch('ser.views.ml_inference.generate_spectrogram_image')
    @patch('ser.views.ml_inference.predict_emotion', side_effect=ValueError('Audio too short'))
    def test_post_short_audio_shows_error(self, mock_predict, mock_spec):
        response = self.client.post(self.url, {'audio_file': _wav_file()})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Audio too short')
        self.assertEqual(PredictionRecord.objects.count(), 0)


class ResultViewTest(TestCase):

    def _make_record(self):
        return PredictionRecord.objects.create(
            emotion='happy',
            confidence=0.92,
            all_probs={'happy': 0.92, 'sad': 0.08},
        )

    def test_get_existing_record_returns_200(self):
        record = self._make_record()
        url = reverse('ser:result', kwargs={'pk': record.pk})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'ser/result.html')

    def test_get_nonexistent_record_returns_404(self):
        url = reverse('ser:result', kwargs={'pk': 9999})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_context_contains_emoji_and_probs(self):
        record = self._make_record()
        url = reverse('ser:result', kwargs={'pk': record.pk})
        response = self.client.get(url)
        self.assertIn('emoji', response.context)
        self.assertEqual(response.context['emoji'], '😄')
        self.assertIn('probs_json', response.context)
        # probs_json must be valid JSON
        parsed = json.loads(response.context['probs_json'])
        self.assertIn('labels', parsed)
        self.assertIn('values', parsed)


class HistoryViewTest(TestCase):

    def test_get_empty_history_returns_200(self):
        response = self.client.get(reverse('ser:history'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'ser/history.html')

    def test_history_shows_existing_records(self):
        PredictionRecord.objects.create(emotion='angry', confidence=0.80)
        PredictionRecord.objects.create(emotion='sad', confidence=0.65)
        response = self.client.get(reverse('ser:history'))
        self.assertEqual(len(response.context['records']), 2)
