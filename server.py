"""
Emotion Detection Server.
Exposes Flask endpoints to analyze emotions from text inputs using
the EmotionDetection package and render the web UI.
"""
from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def emotion_detector_route():
    """
    Analyzes text provided in the 'textToAnalyze' query parameter
    and returns a formatted string with emotion scores and dominant emotion.
    """
    try:
        text_to_analyze = request.args.get("textToAnalyze", "")
        response = emotion_detector(text_to_analyze)

        dominant_emotion = response.get("dominant_emotion") if isinstance(response, dict) else None
        if dominant_emotion is None:
            return "Invalid text! Please try again!"

        return (
            f"For the given statement, the system response is "
            f"'anger': {response.get('anger')}, "
            f"'disgust': {response.get('disgust')}, "
            f"'fear': {response.get('fear')}, "
            f"'joy': {response.get('joy')} and "
            f"'sadness': {response.get('sadness')}. "
            f"The dominant emotion is {dominant_emotion}."
        )
    except Exception:  # pylint: disable=broad-except
        return "Invalid text! Please try again!"


@app.route("/")
def render_index_page():
    """
    Renders the main application HTML page.
    """
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
