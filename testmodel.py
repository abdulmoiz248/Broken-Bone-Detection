import tensorflow as tf
from google.colab import files
from PIL import Image
import numpy as np

# Load models you saved earlier
mobilenetModel = tf.keras.models.load_model('best_mobilenet_model.keras')
deepModel = tf.keras.models.load_model('best_cnn_model.keras')


classNames = ['normal', 'fractured']  
def loadAndPrepareImage(path, imgSize=(128, 128)):
    img = Image.open(path).convert("RGB").resize(imgSize)
    imgArray = np.array(img) / 255.0
    return np.expand_dims(imgArray, axis=0)


uploaded = files.upload()
for fn in uploaded.keys():
    print(f"Uploaded file: {fn}")
    imgInput = loadAndPrepareImage(fn)

    mobilenetPred = mobilenetModel.predict(imgInput)
    mobilenetClass = np.argmax(mobilenetPred, axis=1)[0]
    print("MobileNet predicted class:", classNames[mobilenetClass])

    deepModelPred = deepModel.predict(imgInput)
    deepModelClass = np.argmax(deepModelPred, axis=1)[0]
    print("Custom DL model predicted class:", classNames[deepModelClass])
