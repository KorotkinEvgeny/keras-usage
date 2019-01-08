# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import cv2
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model
from PIL import Image
import numpy as np
import flask
import io

from scipy.misc import imread, imresize
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}


# def load_model():
#     # load the pre-trained Keras model (here we are using a model
#     # pre-trained on ImageNet and provided by Keras, but you can
#     # substitute in your own networks just as easily)
#     global model
#     # model = ResNet50(weights="imagenet")
#     model = load_model()


def prepare_image(image):
    # if the image mode is not RGB, convert it
    # if image.mode != "RGB":
    #     image = image.convert("RGB")

    image = image.resize((28, 28))
    image = image.convert('L')
    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)


    # img = img_to_array(image)
    # img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    # img.reshape(28, 28, 1)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            # image = prepare_image(image, target=(224, 224))
            image = prepare_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image).tolist()
            # print(preds)
            # y_classes = preds.argmax(axis=-1)
            # print(y_classes)
            # predicted_label = label_dict[y_classes[0]]
            # results = imagenet_utils.decode_predictions(preds)
            # data["predictions"] = []
            data["predictions"] = preds

            # loop over the results and add them to the list of
            # returned predictions
            # for (imagenetID, label, prob) in results[0]:
            #     r = {"label": label, "probability": float(prob)}
            #     data["predictions"].append(r)
            #
            # # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    print(data)
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    # load_model()

    model = load_model('cnn4_model.h5')
    model._make_predict_function()
    model.load_weights('cnn4_weights.h5')

    app.run()
