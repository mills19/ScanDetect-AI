from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  #for flash messages
model = load_model('brain_tumor_detection_model.h5')  # Loading model
model.summary()

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocessing
            img = image.load_img(file_path, target_size=(150, 150), color_mode='rgb')
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            
            print(f"Image shape after processing: {img.shape}")

            
            result = model.predict(img)
            prediction = 'Tumor detected' if result[0][0] < 0.5 else 'No tumor detected'

            return render_template('result.html', prediction=prediction)
        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
