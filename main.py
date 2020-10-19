import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import pickle

from skimage import io
from tensorflow import keras

from fastapi import FastAPI, File, HTTPException


app = FastAPI()




path_model = "/content/drive/My Drive/Colab Notebooks/data/final_model"
image_predire = "https://static.toutoupourlechien.com/2017/08/beau-chihuahua.jpg"
image_predire = "https://nosamisleschiens.fr/wp-content/uploads/2017/04/Welsh-Corgi-scaled.jpeg"




IMAGE_SIZE = 150

# METHOD #2: scikit-image

url = image_predire
# download the image using scikit-image

image = io.imread(url)
#cv2.imshow("Incorrect", image)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
from google.colab.patches import cv2_imshow
cv2_imshow(img)

model = keras.models.load_model(path_model)


image = img #img_to_array(img)
image = np.expand_dims(image, axis=0)
# make predictions on the input image
pred = model.predict(image)
pred = pred.argmax(axis=1)[0]


label_encoder = pickle.load(open('/content/drive/My Drive/Colab Notebooks/data_pickle/label_encoder.pickle', 'rb'))


print(pred)


 label_encoder.inverse_transform([pred])


probabilities = model.predict(data_images)

for i in range(10) :
  print(label_encoder.classes_[i])
  print(probabilities[0, i])









def prepare_image(image, target):
    # If the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # Return the processed image
    return image


@app.get("/")
def index():
    return "Hello World!"


@app.post("/predict")
def predict(request: Request, img_file: bytes=File(...)):
    data = {"success": False}

    if request.method == "POST":
        image = Image.open(io.BytesIO(img_file))
        image = prepare_image(image,
                              (int(os.environ.get("IMAGE_WIDTH")),
                               int(os.environ.get("IMAGE_HEIGHT")))
                              )

        # Ensure our NumPy array is C-contiguous as well, otherwise we won't be able to serialize it
        image = image.copy(order="C")

        # Generate an ID for the classification then add the classification ID + image to the queue
        k = str(uuid.uuid4())
        image = base64.b64encode(image).decode("utf-8")
        d = {"id": k, "image": image}
        db.rpush(os.environ.get("IMAGE_QUEUE"), json.dumps(d))

        # Keep looping for CLIENT_MAX_TRIES times
        num_tries = 0
        while num_tries < CLIENT_MAX_TRIES:
            num_tries += 1

            # Attempt to grab the output predictions
            output = db.get(k)

            # Check to see if our model has classified the input image
            if output is not None:
                # Add the output predictions to our data dictionary so we can return it to the client
                output = output.decode("utf-8")
                data["predictions"] = json.loads(output)

                # Delete the result from the database and break from the polling loop
                db.delete(k)
                break

            # Sleep for a small amount to give the model a chance to classify the input image
            time.sleep(float(os.environ.get("CLIENT_SLEEP")))

            # Indicate that the request was a success
            data["success"] = True
        else:
            raise HTTPException(status_code=400, detail="Request failed after {} tries".format(CLIENT_MAX_TRIES))

    # Return the data dictionary as a JSON response
    return data