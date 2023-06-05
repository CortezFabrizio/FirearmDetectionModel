from flask import Flask, request
import tensorflow as tf
from PIL import Image
import json
import os 

app = Flask(__name__)

saved_model_path = os.getenv('MODEL_PATH') 

model = tf.keras.models.load_model(saved_model_path)

@app.route('/firearm_detection', methods=['POST'])
def classify_image():

    if 'image' not in request.files:
        return {"message": "No image provided"}, 400
    
    uploaded_image = request.files['image']

    image = Image.open(uploaded_image).resize((240,240))

    image_to_array = tf.keras.preprocessing.image.img_to_array(image)

    image_dims = tf.expand_dims(image_to_array, axis=0)

    prediction_probability  = (1- model.predict(image_dims)[0][0])*100

    boolean_presence = True if prediction_probability >= 50 else False

    return json.dumps({'firearm_presence':boolean_presence ,"percentage_value": prediction_probability})


if __name__ == '__main__':
    app.run()
