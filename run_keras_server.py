from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
from PIL import Image
import numpy as np
import flask
import io

import tensorflow as tf
graph = tf.get_default_graph()

app = flask.Flask(__name__)
model = None
model_vgg16 = None

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


def vgg16_prepare_image(image):
    img = load_img(image, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


def prepare_image(image):

    image = image.resize((28, 28))
    image = image.convert('L')
    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    return image


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

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
            data["predictions"] = preds

            data["success"] = True

    # return the data dictionary as a JSON response
    print(data)
    return flask.jsonify(data)


@app.route("/pretrained", methods=["POST"])
def pretrained_predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"]

            preprocessed_image = vgg16_prepare_image(image)

            with graph.as_default():
                preds = model_vgg16.predict(preprocessed_image)

            data['predictions'] = str(decode_predictions(preds, top=3)[0])
            data["success"] = True

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
    model_vgg16 = VGG16(weights='imagenet')

    app.run()
