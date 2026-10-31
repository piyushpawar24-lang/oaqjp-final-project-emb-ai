"""
Unit tests for Flask server application in server.py.
"""
import unittest
from unittest.mock import patch
from server import app


class TestFlaskServer(unittest.TestCase):
    """Test suite for validating Flask routes and error responses."""

    def setUp(self):
        """Create a Flask test client before each test."""
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        """Verifies that the root '/' route serves the index.html template successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NLP - Emotion Detection", response.data)
        self.assertIn(b"textToAnalyze", response.data)

    @patch('server.emotion_detector')
    def test_emotion_detector_route_success(self, mock_detector):
        """Verifies '/emotionDetector' formats and returns all emotion scores and dominant emotion."""
        mock_detector.return_value = {
            'anger': 0.01,
            'disgust': 0.02,
            'fear': 0.01,
            'joy': 0.95,
            'sadness': 0.01,
            'dominant_emotion': 'joy'
        }
        response = self.client.get('/emotionDetector?textToAnalyze=I%20am%20glad%20this%20happened')
        self.assertEqual(response.status_code, 200)
        decoded = response.data.decode('utf-8')
        self.assertIn("For the given statement, the system response is", decoded)
        self.assertIn("'anger': 0.01", decoded)
        self.assertIn("'disgust': 0.02", decoded)
        self.assertIn("'fear': 0.01", decoded)
        self.assertIn("'joy': 0.95", decoded)
        self.assertIn("'sadness': 0.01", decoded)
        self.assertIn("The dominant emotion is joy.", decoded)

    def test_emotion_detector_route_empty_param(self):
        """Verifies that an empty query parameter returns the invalid text message."""
        response = self.client.get('/emotionDetector?textToAnalyze=')
        self.assertEqual(response.status_code, 200)
        decoded = response.data.decode('utf-8')
        self.assertEqual(decoded, "Invalid text! Please try again!")

    def test_emotion_detector_route_whitespace_param(self):
        """Verifies that a whitespace-only query parameter returns the invalid text message."""
        response = self.client.get('/emotionDetector?textToAnalyze=%20%20%20')
        self.assertEqual(response.status_code, 200)
        decoded = response.data.decode('utf-8')
        self.assertEqual(decoded, "Invalid text! Please try again!")

    def test_emotion_detector_route_missing_param(self):
        """Verifies that omitting 'textToAnalyze' entirely returns the invalid text message."""
        response = self.client.get('/emotionDetector')
        self.assertEqual(response.status_code, 200)
        decoded = response.data.decode('utf-8')
        self.assertEqual(decoded, "Invalid text! Please try again!")

    @patch('server.emotion_detector')
    def test_emotion_detector_route_detector_failure(self, mock_detector):
        """Verifies that detector failures (all None returned) are handled without crashing."""
        mock_detector.return_value = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
        response = self.client.get('/emotionDetector?textToAnalyze=ServiceFailureScenario')
        self.assertEqual(response.status_code, 200)
        decoded = response.data.decode('utf-8')
        self.assertEqual(decoded, "Invalid text! Please try again!")

    @patch('server.emotion_detector')
    def test_emotion_detector_route_unexpected_exception(self, mock_detector):
        """Verifies that unexpected exceptions do not leak stack traces or crash the server."""
        mock_detector.side_effect = RuntimeError("Unexpected internal crash")
        response = self.client.get('/emotionDetector?textToAnalyze=CrashTrigger')
        self.assertEqual(response.status_code, 200)
        decoded = response.data.decode('utf-8')
        self.assertEqual(decoded, "Invalid text! Please try again!")
        self.assertNotIn("Traceback", decoded)
        self.assertNotIn("RuntimeError", decoded)


if __name__ == '__main__':
    unittest.main()
