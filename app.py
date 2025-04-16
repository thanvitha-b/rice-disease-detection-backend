from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

# Define paths for model files (replace 'YourUsername' with your actual username)
cnn_model_path = "C:/Users/Thanvitha/Downloads/cnn_model.h5"
resnet_model_path = "C:/Users/Thanvitha/Downloads/resnet_model.h5"
mobilenet_model_path = "C:/Users/Thanvitha/Downloads/mobilenet_model.h5"
rf_model_path = "C:/Users/Thanvitha/Downloads/rf_model.pkl"

# Load trained models
cnn_model = tf.keras.models.load_model(cnn_model_path)
resnet_model = tf.keras.models.load_model(resnet_model_path)
mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)
rf_model = pickle.load(open(rf_model_path, 'rb'))

# Define disease classes
disease_classes = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image
        file = request.files['image']
        img = Image.open(file).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using all models
        cnn_pred = cnn_model.predict(img_array)
        resnet_pred = resnet_model.predict(img_array)
        mobilenet_pred = mobilenet_model.predict(img_array)
        rf_pred = rf_model.predict(img_array.reshape(1, -1))

        # Combine predictions (ensemble)
        combined_pred = (cnn_pred + resnet_pred + mobilenet_pred + rf_pred) / 4
        predicted_class = disease_classes[np.argmax(combined_pred)]
        
        # Define prevention measures
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
