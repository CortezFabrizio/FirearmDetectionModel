import json
import os 
from io import BytesIO
from sql_model import Session , Predictions
import boto3

from flask import Flask, request
import tensorflow as tf
from PIL import Image


s3_client = boto3.client('s3')

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

    bool_presence = True if prediction_probability >= 50 else False


    with Session() as session:
        new_prediction =  Predictions(boolean_presence=bool_presence,percentage_probability=prediction_probability)
        session.add(new_prediction)
        session.commit()
        new_prediction_key = new_prediction.key



    image_buffer = BytesIO()
    image.save(image_buffer,format='JPEG')
    s3_client.put_object(
        Bucket='ml-images-project',
        Body=image_buffer.getvalue(),
        Key=f'{str(new_prediction_key)}.jpg'
    )



    return json.dumps({'firearm_presence':bool_presence ,"percentage_value": prediction_probability})



if __name__ == '__main__':
    app.run()

