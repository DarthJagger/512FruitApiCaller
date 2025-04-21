from fastapi import FastAPI, File, UploadFile
import numpy as np # linear algebra
import pandas as pd
import tensorflow as tf
import keras
from keras import Sequential, Model
from keras_preprocessing import image
from keras import optimizers
import tempfile
import shutil
import uvicorn

app = FastAPI()


input_shape = (256, 256, 3) #Cifar10 image size
resized_shape = (224, 224, 3) #EfficientNetV2B0 model input shape
num_classes = 1

# Quadratic confidence; 0.5 = 0 confidence, 0 or 1 = 1 confidence, quadratic scaling inbetween
def getConfidence(prediction):
    return 4 * (prediction - 0.5) ** 2


# Weighted Voting; how confident models are in their predictions, as well as their confidences are taken into account
def weightedVoting(models,image):
    total = 0
    for model in models:
        pred = model.predict(image)[0][0]
        conf = getConfidence(pred)
        if pred >= 0.5:
            total += conf * model.weight # If the model predicts 1, add confidence times weight to total
        else:
            total -= conf * model.weight # If the model predicts 0, subtract confidence times weight from total

    overallPrediction = int(total >= 0) # 0 if 0-voting models were more confident, 1 if 1-voting models were
    return overallPrediction

def buildVGG():
    inputs = keras.layers.Input(shape=(256, 256, 3))
    x = keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224, 3)[:2]))(inputs) #Resize image to  size 224x224
    base_model = keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3), weights=None)
    base_model.trainable = False
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def buildInceptionV3():
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Lambda(lambda image: tf.image.resize(image, resized_shape[:2]))(inputs) #Resize image to  size 224x224
    base_model = keras.applications.InceptionV3(include_top=False, input_shape=resized_shape, weights=None)
    base_model.trainable = False
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class trainedModel:
    def __init__(self, inputName, inputDir, inputValAcc, inputModel):
        self.name = inputName
        self.directory = inputDir
        self.weight = (inputValAcc - 0.5) ** 2 # How much weight to be given to the model, based on test accuracy
        self.model = inputModel
        self.model.load_weights(inputDir)

    def print(self, printStructure = False):
        print("Model name:", self.name)
        print("Model directory:", self.directory)
        print("Model weight:", self.weight)
        if printStructure:
            print("Model structure:")
            self.model.summary()

    def predict(self, img):
        return (self.model.predict(img, verbose = 0))

# Name, directory of the weights, accuracy on test data set, base structure of the network
VGG = trainedModel("VGG", "VGGFinal.keras", 0.64, buildVGG())
InceptionNet = trainedModel("InceptionNet", "Inceptionv3_Model.keras", 0.97, buildInceptionV3())

models = [VGG, InceptionNet]

def predict_file(file):

    test_image = image.load_img(file, target_size=(256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    datagen = image.ImageDataGenerator(rescale=1./224)

    for batch in datagen.flow(test_image, batch_size=1):
        img=batch
        break
    result = weightedVoting(models,img)
    return result

@app.get('/file')
def _file_upload(
        img_file: UploadFile = File(...),
):
    temp_dir = tempfile.mkdtemp()
    file_path = f"{temp_dir}\\{img_file.filename}"
    with open(file_path, "wb") as f:
        f.write(img_file.file.read())

    prediction= predict_file(file_path)
    shutil.rmtree(temp_dir)
    return {
        "prediction":prediction,
    }
#



if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)