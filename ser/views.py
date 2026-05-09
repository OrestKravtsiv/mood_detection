import os
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import ensure_csrf_cookie
from django.conf import settings
from .models import PredictionRecord
from .forms import AudioUploadForm
from . import ml_inference

EMOTION_EMOJI = {
    'angry':     '😡',
    'disgust':   '🤢',
    'fearful':   '😨',
    'happy':     '😄',
    'neutral':   '😐',
    'sad':       '😢',
    'surprised': '😲',
    'calm':      '😌',
}


@ensure_csrf_cookie
def hero_view(request):
    return render(request, 'ser/hero.html')


@require_POST
def predict_api(request):
    form = AudioUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({'error': '; '.join(
            e for errs in form.errors.values() for e in errs
        )}, status=400)

    record = PredictionRecord(audio_file=form.cleaned_data['audio_file'])
    record.save()

    try:
        result = ml_inference.predict_emotion(record.audio_file.path)

        base_name = os.path.splitext(os.path.basename(record.audio_file.name))[0]
        spec_rel  = f'spectrograms/{base_name}.png'
        spec_abs  = os.path.join(settings.MEDIA_ROOT, 'spectrograms', f'{base_name}.png')
        ml_inference.generate_spectrogram_image(result['mel_spec_db'], spec_abs)

        record.emotion    = result['emotion']
        record.confidence = result['confidence']
        record.all_probs  = result['all_probs']
        record.spectrogram = spec_rel
        record.save()
    except ValueError as e:
        record.delete()
        return JsonResponse({'error': str(e)}, status=400)

    result_url = f'/result/{record.pk}/'
    return JsonResponse({
        'emotion':    record.emotion,
        'confidence': record.confidence,
        'all_probs':  record.all_probs,
        'result_url': result_url,
    })


def upload_view(request):
    form = AudioUploadForm()
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            record = PredictionRecord(audio_file=form.cleaned_data['audio_file'])
            record.save()

            try:
                result = ml_inference.predict_emotion(record.audio_file.path)

                base_name = os.path.splitext(os.path.basename(record.audio_file.name))[0]
                spec_rel  = f'spectrograms/{base_name}.png'
                spec_abs  = os.path.join(settings.MEDIA_ROOT, 'spectrograms', f'{base_name}.png')
                ml_inference.generate_spectrogram_image(result['mel_spec_db'], spec_abs)

                record.emotion     = result['emotion']
                record.confidence  = result['confidence']
                record.all_probs   = result['all_probs']
                record.spectrogram = spec_rel
                record.save()
            except ValueError as e:
                record.delete()
                form.add_error('audio_file', str(e))
                return render(request, 'ser/upload.html', {'form': form})

            return redirect('ser:result', pk=record.pk)

    return render(request, 'ser/upload.html', {'form': form})


def result_view(request, pk):
    record = get_object_or_404(PredictionRecord, pk=pk)
    emoji = EMOTION_EMOJI.get(record.emotion, '🎙️')

    probs_json = ''
    if record.all_probs:
        sorted_probs = sorted(record.all_probs.items(), key=lambda x: -x[1])
        probs_json = json.dumps({
            'labels': [p[0] for p in sorted_probs],
            'values': [round(p[1] * 100, 1) for p in sorted_probs],
        })

    return render(request, 'ser/result.html', {
        'record': record,
        'emoji': emoji,
        'probs_json': probs_json,
    })


def history_view(request):
    records = PredictionRecord.objects.order_by('-created_at')[:50]
    return render(request, 'ser/history.html', {'records': records})
