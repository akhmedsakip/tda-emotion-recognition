import numpy as np
import pandas as pd
import librosa as lr
import tensorflow as tf
# Uncomment the line below to disable GPU training if needed
# tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Flatten
from keras.models import Model

FEATURES_PATH = 'features_ts2pi.txt'
LABELS_PATH = 'labels_ts2pi.csv'


loaded = np.loadtxt(FEATURES_PATH)
# Restoring the 3D shape of saved data
X = loaded.reshape(loaded.shape[0], loaded.shape[1] // 224, 224)

scaler = StandardScaler()
# Performing standard scaling on the 3D np.array
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

y = pd.read_csv(LABELS_PATH)['Emotions'].to_numpy()
features = X
labels = y
# Changing the shape of the features to account for the single color channel -> (1440, 224, 224, 1)
features = features.reshape((features.shape[0], features.shape[1], features.shape[2], 1))
features = np.array([np.repeat(img, 3, axis=2) for img in features])


# Create the input layer with the desired input shape
inputs = Input(shape=(224, 224, 3))

# Use a VGG16 model as the base for the new model (not pretrained)
base_model = VGG16(input_tensor=inputs, include_top=False)

# Add a new fully-connected layer with 8 output units
x = base_model.output

# Add a Flatten layer to convert the 7x7x8 tensor output from the VGG16 model to a 7*7*8=392-dimensional vector
x = Flatten()(x)

# To account for 8 emotion classes
x = Dense(8, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# Performing label encoding
labels_old = labels.copy()
le = LabelEncoder()
labels = le.fit_transform(labels_old)

# Basic training code from https://www.kaggle.com/code/nilshmeier/melspectrogram-based-cnn-classification

TEST_SIZE = 0.3
BATCH_SIZE = 64
EPOCHS = 50

encoded_labels = tf.one_hot(indices=labels, depth=8)

X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels.numpy(), test_size=TEST_SIZE)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x=X_train, y=y_train, validation_split=TEST_SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping])

y_predicted = np.argmax(model.predict(x=X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy_score(y_true, y_predicted)