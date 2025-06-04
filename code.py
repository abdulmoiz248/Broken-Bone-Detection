# ✅ SETUP
#!pip install roboflow --quiet
from roboflow import Roboflow
rf = Roboflow(api_key="")  # 🔐 replace with your Roboflow key
project = rf.workspace("yolo-mqm4e").project("joint-bone-fracture-shuffled")
dataset = project.version(1).download("folder")

# ✅ IMPORTS
import os, gc, cv2, numpy as np, matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# ✅ LOAD IMAGES
def load_images(path, label, imgSize=(128, 128)):
    data, labels = [], []
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(path, file))
            img = cv2.resize(img, imgSize)
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

X, y = [], []
for className in os.listdir(dataset.location + "/train"):
    imgs, labels = load_images(os.path.join(dataset.location, "train", className), className)
    X.append(imgs)
    y.append(labels)

X = np.concatenate(X)
y = np.concatenate(y)

# Preprocessing labels
labelEncoder = LabelEncoder()
yEnc = labelEncoder.fit_transform(y)

# Split dataset
XTrain, XTest, yTrain, yTest = train_test_split(X, yEnc, test_size=0.2, random_state=42)

# Flatten images for KNN and SVM
XTrainFlat = XTrain.reshape((XTrain.shape[0], -1))
XTestFlat = XTest.reshape((XTest.shape[0], -1))

# KNN
knn = KNeighborsClassifier()
knn.fit(XTrainFlat, yTrain)
knnPred = knn.predict(XTestFlat)
print(f"KNN Accuracy: {accuracy_score(yTest, knnPred):.4f}")

# SVM
svm = SVC(kernel='linear')
svm.fit(XTrainFlat, yTrain)
svmPred = svm.predict(XTestFlat)
print(f"SVM Accuracy: {accuracy_score(yTest, svmPred):.4f}")

# Prepare labels for NN (one-hot)
yTrainEnc = tf.keras.utils.to_categorical(yTrain)
yTestEnc = tf.keras.utils.to_categorical(yTest)

# CNN Model
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=XTrain.shape[1:]),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(yTrainEnc.shape[1], activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnnChk = ModelCheckpoint("best_cnn_model.keras", save_best_only=True, monitor='val_accuracy', mode='max')

print("\nTraining CNN Model:")
cnnHistory = cnn.fit(
    XTrain, yTrainEnc,
    epochs=5,
    validation_data=(XTest, yTestEnc),
    callbacks=[cnnChk],
    verbose=1
)

print(f"Best CNN val_accuracy: {max(cnnHistory.history['val_accuracy']):.4f}")

# MobileNetV2 Model
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=XTrain.shape[1:])
baseModel.trainable = False

mobilenet = Sequential([
    baseModel,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(yTrainEnc.shape[1], activation='softmax')
])

mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mobileChk = ModelCheckpoint("best_mobilenet_model.keras", save_best_only=True, monitor='val_accuracy', mode='max')

print("\nTraining MobileNetV2 Model:")
mobileHistory = mobilenet.fit(
    preprocess_input(XTrain), yTrainEnc,
    epochs=5,
    validation_data=(preprocess_input(XTest), yTestEnc),
    callbacks=[mobileChk],
    verbose=1
)

print(f"Best MobileNet val_accuracy: {max(mobileHistory.history['val_accuracy']):.4f}")

# Done. Models saved as best_cnn_model.keras and best_mobilenet_model.keras
