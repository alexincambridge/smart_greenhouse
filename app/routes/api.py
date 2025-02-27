from flask import Blueprint, request, jsonify
import ml_model
import camera

api_bp = Blueprint("api", __name__)

@api_bp.route("/detect", methods=["POST"])
def detect_pest():
    """API to capture image & detect pests."""
    camera.capture_and_detect()
    result = ml_model.predict_pest("data/images/live.jpg")
    return jsonify({"status": "success", "result": result})

