import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
MODEL_PATH = os.path.join('model', 'model.h5')
model = load_model(MODEL_PATH)

# Class labels (replace with your PlantVillage labels)
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust_",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy"
    # Add all other classes as per your dataset
]

# Disease remedies dictionary
disease_remedies = {
    "Apple___Apple_scab": "Remove infected leaves, use sulfur sprays, ensure good airflow.",
    "Apple___Black_rot": "Prune affected areas, apply fungicide, rotate crops.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees, use resistant varieties, fungicide application.",
    "Apple___healthy": "No action needed. Maintain proper care.",
    "Blueberry___healthy": "No action needed. Maintain soil moisture and sunlight.",
    "Cherry___Powdery_mildew": "Spray neem oil, remove affected leaves, increase sunlight exposure.",
    "Cherry___healthy": "No action needed. Regular watering and pruning.",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Remove infected leaves, crop rotation, fungicide treatment.",
    "Corn___Common_rust_": "Use resistant varieties, fungicides, and monitor humidity.",
    "Corn___Northern_Leaf_Blight": "Remove infected debris, apply fungicide early.",
    "Corn___healthy": "Maintain proper watering and nutrients."
    # Add all other classes
}
# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # normalize
        
        # Predict
        preds = model.predict(x)
        predicted_class = np.argmax(preds, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        remedy_text = disease_remedies.get(predicted_label, "No remedy found for this disease.")
        
        return render_template('result.html', filename=filename, prediction=predicted_label, remedy=remedy_text)

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
