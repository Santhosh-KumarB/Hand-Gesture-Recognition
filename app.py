from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("hand_gesture_model.h5")

# Gesture labels (VERY IMPORTANT)
gesture_labels = {
    0: "Palm",
    1: "L",
    2: "Fist",
    3: "Fist Moved",
    4: "Thumb",
    5: "Index",
    6: "OK",
    7: "Palm Moved",
    8: "C",
    9: "Down"
}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # Save uploaded image
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # Load image, convert to GRAYSCALE (FIX)
            img = Image.open(image_path).convert("L")
            img = img.resize((64, 64))

            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 64, 64, 1)

            # Predict
            prediction = model.predict(img_array)
            class_index = int(np.argmax(prediction))

            # Get label safely
            prediction_text = gesture_labels.get(class_index, "Unknown Gesture")

        else:
            prediction_text = "Please upload a JPG or PNG image only."

    return render_template(
        "index.html",
        prediction=prediction_text,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
