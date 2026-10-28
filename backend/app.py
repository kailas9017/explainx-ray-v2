import base64
import os
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

from model_service import analyze_image_bytes, ModelNotReadyError, MODEL_STATUS

app = Flask(__name__)
CORS(app)  # allow the Vite dev server (different port) to call this API


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(MODEL_STATUS)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("imageBase64")

    if not image_b64:
        return jsonify({"error": "Missing imageBase64 field"}), 400

    try:
        clean_b64 = image_b64.split(",")[-1]  # strip "data:image/png;base64," prefix if present
        image_bytes = base64.b64decode(clean_b64)
    except Exception:
        return jsonify({"error": "imageBase64 could not be decoded"}), 400

    try:
        result = analyze_image_bytes(image_bytes)
        return jsonify(result)
    except ModelNotReadyError as e:
        return jsonify({"error": "Model not trained yet", "details": str(e)}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Inference failed", "details": str(e)}), 500


if __name__ == "__main__":
    # Single process, model loaded once at import time (see model_service.py).
    # debug=False on purpose: the reloader would re-import model_service.py
    # (and re-load the model onto the GPU) on every file save.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
