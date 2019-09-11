from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from keras.models import load_model

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, MobileNet,decode_predictions
import numpy as np

model = load_model('Mobinet.h5')

import numpy as np
import tensorflow as tf
graph = tf.get_default_graph()

app = Flask(__name__)
# Enable CORS
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
  result = 0
  if request.method == "POST":
    
    img_path = './elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    with graph.as_default():
      features = model.predict(x)
    result = str(features.argmax(axis=1)[0])

  return jsonify(
    prediction=result
  ),201