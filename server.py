from flask import Flask, make_response, request, jsonify, render_template, flash, redirect
import numpy as np
from werkzeug.utils import secure_filename
from keras.models import load_model
import keras
import librosa
import os

CLASSES = ["Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", "Drilling","Engine Idling","Gun Shot","Jackhammer","Siren","Street Music"]
IMAGES = {
    'Air Conditioner' : 'https://media.kasperskydaily.com/wp-content/uploads/sites/92/2016/02/06022934/ac-vs-powergrid-featured.jpg',
    'Car Horn' : 'https://daily.jstor.org/wp-content/uploads/2017/12/traffic_jam_1050x700.jpg',
    'Children Playing' : 'https://content.health.harvard.edu/wp-content/uploads/2023/04/fef23683-0b70-49d7-8b32-d11ae842f633.jpg',
    'Dog Bark' : 'https://windycitypaws.com/wp-content/uploads/2016/02/Two-dogs-with-leash-e1483989040950.jpg',
    'Drilling' : 'https://l450v.alamy.com/450v/2h2p27j/two-african-american-construction-workers-safely-drilling-hole-in-city-center-adjacent-to-wisconsin-river-urban-renewal-milwaukee-wisconsin-usa-2h2p27j.jpg',
    'Engine Idling': 'https://www.timeforkids.com/wp-content/uploads/2020/04/TFK_200417_035-e1586193133150.jpg',
    'Gun Shot' : 'https://i.insider.com/59dbf7fd6d80adac108b50e7?width=1010&format=jpeg',
    'Jackhammer' : 'https://i5.walmartimages.com/asr/78fe03dc-7270-4605-a4c9-f5b75e26442f.01280232a103491c4fe1f387cb357a22.jpeg',
    'Siren' : 'https://content.ucpress.edu/blog/wp-content/uploads/2020/04/Seim.jpg',
    'Street Music' : 'https://upload.wikimedia.org/wikipedia/commons/f/f4/One-man_band_street_performer_-_5.jpg'
    }
model_path = './cnn.keras'
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

model = load_model(model_path)
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

def allowed_file(filename):
    # Boolean helper to check if file is valid mp3
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload')
def upload_file():
   return render_template('./upload.html')

@app.route('/uploader', methods = ['GET','POST'])
def upload_file_and_predict():
   # Handles user upload, and returns the predicted class
   if request.method == 'POST':
      # check if the HTTP POST request has the 'file' field
        if 'file' not in request.files:
            flash('No file in HTTP request.')
            return redirect(request.url)
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return 'Bad request', 400
        
        if allowed_file(f.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            print('file uploaded successfully')
            return get_features_and_predict(filepath)
        else:
            return 'Bad request', 400

   if request.method == 'GET':
        #flash('No selected file')
        return 'No selected file', 400
   
   return 'Bad request', 400

def get_features_and_predict(path):
    # Load uploaded mp3 
    data, sample_rate = librosa.load(path, res_type='fft')

    # Feature extraction
    ft = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=128).T
    ft = np.mean(ft, axis=0)    
    ft = ft.reshape((1,16,4,2))   

    # Run prediction and return results
    predict_results = model.predict(ft[:1])
    probabilities = predict_results * 100
    prediction = CLASSES[np.argmax(predict_results)]
    response = jsonify(predictions = prob_dict(probabilities=probabilities[0]), predicted_class = prediction, img_url = IMAGES[prediction], code='SUCCESS')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/live', methods = ['GET'])
def api_health_check():
    return {"STATUS": "ALL SYSTEMS NOMINAL"}

def prob_dict(probabilities):
    return {CLASSES[i]: str(probabilities[i]) for i in range(len(CLASSES))}
        
if __name__ == '__main__':
    app.run(port=5000, debug=True)