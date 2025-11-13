import os
import pickle
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Flask app setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load model
# model = tf.keras.models.load_model("model/cat_dog_mobilenetv2.h5")
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
model = tf.keras.models.load_model(
    "model/cat_dog_mobilenetv2.h5",
    custom_objects={'preprocess_input': preprocess_input}
)


# Load class names
with open("model/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    predicted_class = class_names[class_index]

    return predicted_class

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected"

        # Save uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Predict
        result = predict_image(file_path)

        return render_template("result.html", prediction=result, img_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
