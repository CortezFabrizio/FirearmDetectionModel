import os
from PIL import Image
import tensorflow as tf

# FOR CUSTOM TESTING AFTER TRAINING

saved_model_path = os.getenv('MODEL_PATH') 

model = tf.keras.models.load_model(saved_model_path)

folder_with_images_path = os.getenv('TESTING_IMAGES')

list_images = os.listdir(folder_with_images_path)

for image_name in list_images:

    if image_name.endswith('.jpg'):

        image_path = os.path.join(folder_with_images_path, image_name)

        image = Image.open(image_path).resize((240,240))

        image.show()

        image_to_array = tf.keras.preprocessing.image.img_to_array(image)

        image_dims = tf.expand_dims(image_to_array, axis=0)

        prediction = model.predict(image_dims)[0][0]

        print('Probability of presence of a firearm: %',(1 - prediction)*100,image_name)


