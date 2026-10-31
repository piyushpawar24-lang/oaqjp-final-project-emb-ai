"""
Compatibility forwarding module for root-level emotion_detection.
Delegates implementation to the packaged EmotionDetection.emotion_detection module.
"""
from EmotionDetection.emotion_detection import emotion_detector

__all__ = ['emotion_detector']
