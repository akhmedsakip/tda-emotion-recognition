import numpy as np
import pandas as pd
import librosa as lr
import gudhi
from multiprocessing import Pool
import math
import tqdm

AUDIO_PATHS = 'audio_paths.csv'
# Modify the below variable depending on the number of CPU threads you are ready to allocate
N_THREADS = 16


# Extracting persistence barcode given the path of an audio file
# This code is significantly changed based on the original code from https://github.com/mltlachac/TDA
def extract_persistence_barcode(path):
    data, sr = lr.load(path, sr=16000, duration=2.5, offset=0.6)
    simplex = gudhi.SimplexTree()
    for i in np.arange(len(data)):
        simplex.insert([i, i+1], filtration = data[i])
    barcode = simplex.persistence()
    return barcode

# Takes a persistence barcode as input, and iterates over each of its inner lists. If a value in an inner list
# is equal to positive infinity, the function replaces it with a different value and appends the resulting inner list
# to another list.
def format_pd(persistence_barcode):
    res_array = []
    for row in persistence_barcode:
        push_val_0 = row[1][0]
        push_val_1 = row[1][1]
        if math.isinf(row[1][0]):
            push_val_0 = -0.2
        if math.isinf(row[1][1]):
            push_val_1 = 0.2
        res_array.append([push_val_0, push_val_1])
    return res_array

# Parallelizing the feature extraction function to speed up the process
def preprocess_dataset(df):
    with Pool(N_THREADS) as p:
        X = p.map(extract_persistence_barcode, df['Path'])

    with Pool(N_THREADS) as p:
        X_numpy = p.map(format_pd, X)
    
    # Calculating bottleneck distances for every pair of persistence diagrams using gudhi
    # WARNING: This functions runs for 2+ hours!
    total_distances = []
    for pd1 in tqdm.tqdm(X_numpy):
        diagram_distances = []
        for pd2 in X_numpy:
            diagram_distances.append(gudhi.bottleneck_distance(pd1, pd2, 0.1))
        total_distances.append(diagram_distances)

    y = df['Emotions']

    return total_distances, y


audio_paths = pd.read_csv(AUDIO_PATHS)
X, y = preprocess_dataset(audio_paths)

# Saving the preprocessed features (pairwise distances) for future use
preprocessed = pd.DataFrame(X)
preprocessed['label'] = y
preprocessed.to_csv('features_pairwise.csv', index=False)