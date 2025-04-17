from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Define paths for model files
cnn_model_path = r"C:\Users\boyap\Downloads\cnn_model.h5"
resnet_model_path = r"C:\Users\boyap\Downloads\resnet_model.h5"
mobilenet_model_path = r"C:\Users\boyap\Downloads\mobilenet_model.h5"
rf_model_path = r"C:\Users\boyap\Downloads\rf_model.pkl"

# Load trained models
cnn_model = tf.keras.models.load_model(cnn_model_path)
resnet_model = tf.keras.models.load_model(resnet_model_path)
mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)
with open(rf_model_path, 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

# Define disease classes
disease_classes = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the 'image' key is in the uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded. Please upload an image.'}), 400

        # Load and preprocess the uploaded image
        file = request.files['image']
        img = Image.open(file).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predictions from each model
        cnn_pred = cnn_model.predict(img_array)
        resnet_pred = resnet_model.predict(img_array)
        mobilenet_pred = mobilenet_model.predict(img_array)
        rf_pred = rf_model.predict(img_array.reshape(1, -1))

        # Ensemble prediction: average predictions
        combined_pred = (cnn_pred + resnet_pred + mobilenet_pred) / 3  # Remove `rf_pred` as it might not fit seamlessly
        predicted_class = disease_classes[np.argmax(combined_pred)]

        # Prevention measures
        prevention_measures = {
            'Bacterial Leaf Blight': 'Ensure proper drainage. Use resistant varieties.',
            'Brown Spot': 'Apply fungicides. Maintain balanced nitrogen levels.',
            'Healthy': 'No action needed. Keep monitoring crops.',
            'Leaf Blast': 'Avoid excess nitrogen. Apply protective fungicides.',
            'Leaf Scald': 'Improve water management. Use disease-resistant varieties.',
            'Narrow Brown Spot': 'Apply potassium-rich fertilizers. Use fungicides.'
        }

        return jsonify({
            'disease': predicted_class,
            'prevention': prevention_measures[predicted_class]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
