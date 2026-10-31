"""
Unit tests for Task 3: Output Formatting in emotion_detection.py.
"""
import json
import unittest
from unittest.mock import Mock, patch
import requests

from EmotionDetection.emotion_detection import emotion_detector


class TestTask3Formatting(unittest.TestCase):
    """Test suite for verifying emotion_detector output formatting and error handling."""

    def test_1_joy_dominant(self):
        """Test 1: Successful response where joy has the highest score."""
        mock_payload = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.01,
                    'disgust': 0.02,
                    'fear': 0.03,
                    'joy': 0.89,
                    'sadness': 0.05
                }
            }]
        }
        mock_resp = Mock(status_code=200, text=json.dumps(mock_payload))
        with patch('requests.post', return_value=mock_resp):
            result = emotion_detector("I am glad this happened")
            self.assertEqual(result['dominant_emotion'], 'joy')

    def test_2_sadness_dominant(self):
        """Test 2: Successful response where sadness has the highest score."""
        mock_payload = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.05,
                    'disgust': 0.02,
                    'fear': 0.04,
                    'joy': 0.01,
                    'sadness': 0.88
                }
            }]
        }
        mock_resp = Mock(status_code=200, text=json.dumps(mock_payload))
        with patch('requests.post', return_value=mock_resp):
            result = emotion_detector("I am so sad about this")
            self.assertEqual(result['dominant_emotion'], 'sadness')

    def test_3_extraction_of_all_five_scores(self):
        """Test 3: Correct extraction of all five emotion scores."""
        mock_payload = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.123,
                    'disgust': 0.234,
                    'fear': 0.345,
                    'joy': 0.456,
                    'sadness': 0.567
                }
            }]
        }
        mock_resp = Mock(status_code=200, text=json.dumps(mock_payload))
        with patch('requests.post', return_value=mock_resp):
            result = emotion_detector("Sample text")
            self.assertEqual(result['anger'], 0.123)
            self.assertEqual(result['disgust'], 0.234)
            self.assertEqual(result['fear'], 0.345)
            self.assertEqual(result['joy'], 0.456)
            self.assertEqual(result['sadness'], 0.567)

    def test_4_dominant_emotion_calculation(self):
        """Test 4: Correct calculation of dominant_emotion when anger is highest."""
        mock_payload = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.91,
                    'disgust': 0.03,
                    'fear': 0.02,
                    'joy': 0.01,
                    'sadness': 0.03
                }
            }]
        }
        mock_resp = Mock(status_code=200, text=json.dumps(mock_payload))
        with patch('requests.post', return_value=mock_resp):
            result = emotion_detector("I am furious")
            self.assertEqual(result['dominant_emotion'], 'anger')

    def test_5_unsuccessful_http_response(self):
        """Test 5: An unsuccessful HTTP response (e.g. 400 or 500)."""
        mock_resp = Mock(status_code=400, text='{"error": "bad request"}')
        with patch('requests.post', return_value=mock_resp):
            result = emotion_detector("Invalid input statement")
            expected = {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }
            self.assertEqual(result, expected)

    def test_6_connection_timeout(self):
        """Test 6: A connection timeout / network failure."""
        with patch('requests.post', side_effect=requests.exceptions.ConnectTimeout("Timed out")):
            result = emotion_detector("I am glad this happened")
            expected = {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }
            self.assertEqual(result, expected)

    def test_7_malformed_json_or_missing_fields(self):
        """Test 7: Malformed JSON or missing emotion fields in response."""
        # Case A: Corrupted JSON
        mock_resp_corrupted = Mock(status_code=200, text="{broken_json")
        with patch('requests.post', return_value=mock_resp_corrupted):
            result_a = emotion_detector("Test text")
            self.assertIsNone(result_a['dominant_emotion'])

        # Case B: Missing emotionPredictions key
        mock_resp_empty = Mock(status_code=200, text=json.dumps({"wrongKey": []}))
        with patch('requests.post', return_value=mock_resp_empty):
            result_b = emotion_detector("Test text")
            self.assertIsNone(result_b['dominant_emotion'])

    def test_8_empty_input(self):
        """Test 8: Empty or whitespace-only input."""
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
        self.assertEqual(emotion_detector(None), expected)


if __name__ == '__main__':
    unittest.main()

