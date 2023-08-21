# urban-sound-classify-api
A Flask API serving a machine learning model that classifies urban sounds.  
Upon receiving API requests at `uploader` containing .mp3 or .wav file, the server will run the file through the trained keras model with the model outputs and respond with the predicted class.

## How to Set Up
1. Clone repo and install required packages — Librosa, Tensorflow, Keras, Flask
2. In the root directory, run command `python server.py`
3. The server is now running and able to respond to HTTP requests at `/uploader` endpoint.
