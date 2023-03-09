import numpy as np
import pandas as pd
import librosa as lr
from ts2pi import persistence_image
from multiprocessing import Pool

AUDIO_PATHS = 'audio_paths.csv'
# Modify the below variable depending on the number of CPU threads you are ready to allocate
N_THREADS = 16
IMAGE_RESOLUTION = [224, 224]

# Extracting 224x224 persistence images for the provided path of an audio file
def extract_features(path):
    wave, sample_rate = lr.core.load(path)

    image = persistence_image(wave, resolution=IMAGE_RESOLUTION, t=3, bandwidth=0.1)
    
    return image

# Iterating through the whole dataframe and extracting persistence images for each audio
def preprocess_dataset(df):
    X = []
    i = 0
    for path in df['Path']:
        feature = extract_features(path)
        X.append(feature)
        if i % 10 == 0:
            print("Extracted", i, "-th feature")
        i += 1
    
    y = df['Emotions']

    return X, y

audio_paths = pd.read_csv(AUDIO_PATHS)
X, y = preprocess_dataset(audio_paths)

X_np = np.array(X)
# Reshaping X so that it can be saved in a text file (since it is a 3D np.array)
X_reshaped = X_np.reshape(X_np.shape[0], -1)
np.savetxt("features_ts2pi.txt", X_reshaped)
# loading this file
# loaded = np.loadtxt("features_ts2pi.txt")
# X = loaded.reshape(loaded.shape[0], loaded.shape[1] // 224, 224)

# Separately storing the corresponding emotion labels
y.to_csv("labels_ts2pi.csv", index=False)