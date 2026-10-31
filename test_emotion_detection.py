"""
Unit tests for the EmotionDetection package.
Verifies emotion prediction, dominant emotion identification, API contracts,
and comprehensive error handling.
"""
import json
import unittest
from unittest.mock import Mock, patch
import requests

from EmotionDetection.emotion_detection import emotion_detector


def _mock_watson_api(*args, **kwargs):
    """
    Simulates Watson NLP EmotionPredict responses for known benchmark statements
    and standard inputs without requiring live cloud connectivity.
    """
    payload = kwargs.get('json', {})
    text = payload.get('raw_document', {}).get('text', '')

    if 'glad' in text:
        emotions = {'anger': 0.01, 'disgust': 0.02, 'fear': 0.01, 'joy': 0.95, 'sadness': 0.01}
    elif 'mad' in text:
        emotions = {'anger': 0.92, 'disgust': 0.03, 'fear': 0.02, 'joy': 0.01, 'sadness': 0.02}
    elif 'disgusted' in text:
        emotions = {'anger': 0.04, 'disgust': 0.91, 'fear': 0.02, 'joy': 0.01, 'sadness': 0.02}
    elif 'sad' in text:
        emotions = {'anger': 0.02, 'disgust': 0.01, 'fear': 0.03, 'joy': 0.01, 'sadness': 0.93}
    elif 'afraid' in text:
        emotions = {'anger': 0.02, 'disgust': 0.01, 'fear': 0.94, 'joy': 0.01, 'sadness': 0.02}
    else:
        emotions = {'anger': 0.1, 'disgust': 0.1, 'fear': 0.1, 'joy': 0.5, 'sadness': 0.1}

    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = json.dumps({'emotionPredictions': [{'emotion': emotions}]})
    return mock_resp


class TestEmotionDetection(unittest.TestCase):
    """Test suite covering assignment requirements and comprehensive error handling."""

    @patch('EmotionDetection.emotion_detection.requests.post', side_effect=_mock_watson_api)
    def test_emotion_detector(self, mock_post):
        """
        Tests the 5 official assignment statements and verifies their dominant emotions:
        1. 'I am glad this happened' -> joy
        2. 'I am really mad about this' -> anger
        3. 'I feel disgusted just hearing about this' -> disgust
        4. 'I am so sad about this' -> sadness
        5. 'I am really afraid that this will happen' -> fear
        """
        # Test 1: joy
        result_1 = emotion_detector("I am glad this happened")
        self.assertEqual(result_1['dominant_emotion'], 'joy')

        # Test 2: anger
        result_2 = emotion_detector("I am really mad about this")
        self.assertEqual(result_2['dominant_emotion'], 'anger')

        # Test 3: disgust
        result_3 = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result_3['dominant_emotion'], 'disgust')

        # Test 4: sadness
        result_4 = emotion_detector("I am so sad about this")
        self.assertEqual(result_4['dominant_emotion'], 'sadness')

        # Test 5: fear
        result_5 = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result_5['dominant_emotion'], 'fear')

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_request_contract(self, mock_post):
        """Verifies that the correct Watson NLP endpoint, header, and payload are sent."""
        mock_post.return_value = Mock(
            status_code=200,
            text=json.dumps({
                'emotionPredictions': [{
                    'emotion': {'anger': 0.1, 'disgust': 0.1, 'fear': 0.1, 'joy': 0.5, 'sadness': 0.1}
                }]
            })
        )
        test_statement = "Testing contract payload"
        emotion_detector(test_statement)

        mock_post.assert_called_once()
        expected_url = (
            'https://sn-watson-emotion.labs.skills.network/'
            'v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
        )
        called_url = mock_post.call_args[0][0]
        called_headers = mock_post.call_args[1]['headers']
        called_payload = mock_post.call_args[1]['json']

        self.assertEqual(called_url, expected_url)
        self.assertEqual(
            called_headers.get("grpc-metadata-mm-model-id"),
            "emotion_aggregated-workflow_lang_en_stock"
        )
        self.assertEqual(called_payload, {"raw_document": {"text": test_statement}})

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_all_five_scores_extracted(self, mock_post):
        """Verifies that all 5 emotion scores are extracted and present in the returned dictionary."""
        mock_post.return_value = Mock(
            status_code=200,
            text=json.dumps({
                'emotionPredictions': [{
                    'emotion': {
                        'anger': 0.11,
                        'disgust': 0.22,
                        'fear': 0.33,
                        'joy': 0.44,
                        'sadness': 0.55
                    }
                }]
            })
        )
        result = emotion_detector("Valid statement")
        self.assertEqual(result['anger'], 0.11)
        self.assertEqual(result['disgust'], 0.22)
        self.assertEqual(result['fear'], 0.33)
        self.assertEqual(result['joy'], 0.44)
        self.assertEqual(result['sadness'], 0.55)
        self.assertEqual(result['dominant_emotion'], 'sadness')

    def test_invalid_and_empty_input(self):
        """Verifies that empty string, whitespace-only, None, and non-string inputs return all None."""
        expected = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
        self.assertEqual(emotion_detector(""), expected)
        self.assertEqual(emotion_detector("   "), expected)
        self.assertEqual(emotion_detector("\t\n"), expected)
        self.assertEqual(emotion_detector(None), expected)
        self.assertEqual(emotion_detector(12345), expected)
        self.assertEqual(emotion_detector(['some', 'list']), expected)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_http_error_handling(self, mock_post):
        """Verifies that HTTP 400, 404, 500, and 503 return the default None dictionary."""
        expected = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
        for code in [400, 404, 500, 503]:
            mock_post.return_value = Mock(status_code=code, text='{"error": "API failure"}')
            self.assertEqual(emotion_detector("Error testing"), expected)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_connection_timeout_and_network_error(self, mock_post):
        """Verifies that connection timeouts and network errors return the default None dictionary."""
        expected = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
        mock_post.side_effect = requests.exceptions.ConnectTimeout("Connection timed out")
        self.assertEqual(emotion_detector("Timeout statement"), expected)

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        self.assertEqual(emotion_detector("Connection error statement"), expected)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_malformed_json_and_missing_fields(self, mock_post):
        """Verifies that corrupted JSON, missing keys, and empty dictionaries return all None."""
        expected = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
        # Broken JSON string
        mock_post.return_value = Mock(status_code=200, text="{broken-json")
        self.assertEqual(emotion_detector("Malformed"), expected)

        # Missing emotionPredictions key
        mock_post.return_value = Mock(status_code=200, text=json.dumps({"unrelated": 123}))
        self.assertEqual(emotion_detector("Missing key"), expected)

        # Empty emotionPredictions list
        mock_post.return_value = Mock(status_code=200, text=json.dumps({"emotionPredictions": []}))
        self.assertEqual(emotion_detector("Empty predictions"), expected)

        # Missing 'emotion' key in prediction
        mock_post.return_value = Mock(status_code=200, text=json.dumps({"emotionPredictions": [{}]}))
        self.assertEqual(emotion_detector("Missing emotion key"), expected)

        # Empty 'emotion' dictionary
        mock_post.return_value = Mock(
            status_code=200,
            text=json.dumps({"emotionPredictions": [{"emotion": {}}]})
        )
        self.assertEqual(emotion_detector("Empty emotion dict"), expected)


if __name__ == '__main__':
    unittest.main()
