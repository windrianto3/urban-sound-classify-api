# urban-sound-classify-api
A Flask API serving a machine learning model that classifies urban sounds.  
Upon receiving API requests at `uploader` containing .mp3 or .wav file, the server will run the file through the trained keras model with the model outputs and respond with the predicted class.

## Components
### cnn.keras
This is the machine learning model that takes in sound features and returns a predicted class out of 10 possibilities (`air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music`)
### main.py
This is the Flask API that serves the model. The `uploader` endpoint takes POST requests with a single MP3 or WAV file encompassed as FormData, then performs feature extraction + classification, and finally returns the predicted class.

## How to Set Up
1. Clone repo and install required packages — Librosa, Tensorflow, Keras, Flask
2. In the root directory, run command `python server.py`
3. The server is now running and able to respond to HTTP requests at `/uploader` endpoint.

## Additional Info
I created a UI for this Flask application using React. You can view it [here](https://github.com/windrianto3/urban-sound-classify-ui).
