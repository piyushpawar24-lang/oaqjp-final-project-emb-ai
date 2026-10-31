"""
Emotion Detection module using IBM Watson NLP library.
"""
import json
import requests


def emotion_detector(text_to_analyze):
    """
    Sends a text analysis request to the Watson NLP EmotionPredict service,
    extracts emotion scores for anger, disgust, fear, joy, and sadness,
    identifies the dominant emotion, and returns the formatted dictionary.
    """
    failed_response = {
        'anger': None,
        'disgust': None,
        'fear': None,
        'joy': None,
        'sadness': None,
        'dominant_emotion': None
    }

    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return failed_response

    url = (
        'https://sn-watson-emotion.labs.skills.network/'
        'v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            formatted_response = json.loads(response.text)
            predictions = formatted_response.get('emotionPredictions', [])

            if predictions and isinstance(predictions[0], dict):
                emotions = predictions[0].get('emotion', {})

                if emotions and isinstance(emotions, dict):
                    keys = ['anger', 'disgust', 'fear', 'joy', 'sadness']
                    scores = {k: float(emotions.get(k, 0.0)) for k in keys}
                    scores['dominant_emotion'] = max(scores, key=scores.get)
                    return scores

    except (requests.exceptions.RequestException, json.JSONDecodeError,
            KeyError, TypeError, ValueError):
        pass

    return failed_response
