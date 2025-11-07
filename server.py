from flask import Flask, request, jsonify
from waste_model import WasteClassifier
from PIL import Image
import os

app = Flask(__name__)
model = WasteClassifier()

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = "/tmp/uploaded.jpg"
    image.save(image_path)

    result = model.classify(image_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
