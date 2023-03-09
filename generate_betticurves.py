import numpy as np
import pandas as pd
import librosa as lr
import gudhi
from multiprocessing import Pool

AUDIO_PATHS = 'audio_paths.csv'
BETTI_COMPONENTS = 100
# Modify the below variable depending on the number of CPU threads you are ready to allocate
N_THREADS = 16

# Extracting persistence barcode from an audio wave extracted using librosa
# The code is used from https://github.com/mltlachac/TDA with no significant changes
def get_persistence_from_audio(audio_wave, sample=22050, length=5):
    simplex_up = gudhi.SimplexTree()
    simplex_dw = gudhi.SimplexTree()
    for i in np.arange(len(audio_wave)):
        simplex_up.insert([i, i + 1], filtration=audio_wave[i])
        simplex_dw.insert([i, i + 1], filtration=-audio_wave[i])
    for i in np.arange(len(audio_wave) - 1):
        simplex_up.insert([i, i + 1], filtration=audio_wave[i])
        simplex_dw.insert([i, i + 1], filtration=-audio_wave[i])
    dig_up = simplex_up.persistence()
    dig_dw = simplex_dw.persistence()
    return dig_up, dig_dw

# A helper function
# The code is used from https://github.com/mltlachac/TDA with no significant changes
def functionize(val, descriptor):
    def dirichlet(x):
        return 1 if (x > descriptor[0]) and (x < descriptor[1]) else 0

    return np.vectorize(dirichlet)(val)

# Extracting Betti curves explained by 100 points given a persistence barocde of an audio file
# The code is used from https://github.com/mltlachac/TDA with no significant changes
def get_betti_curve_from_persistence(dig, num_points=100):
    dig = np.asarray([[ele[1][0], ele[1][1]] for ele in dig if ele[1][1] < np.inf])
    v = np.zeros(num_points)
    try:
        mn, mx = np.min(dig), np.max(dig)
        val_up = np.linspace(mn, mx, num=num_points)
        for ele in dig:
            v += functionize(val_up, ele)
        return v
    except ValueError:
        print("Silent, returning 0")
        return v

# Extracting the resulting features (Betti curves) using the above functions
def extract_features(path):
    wave, sample_rate = lr.core.load(path)

    dig_up, dig_dw = get_persistence_from_audio(wave, sample=sample_rate)
    betti = get_betti_curve_from_persistence(dig_dw, num_points=BETTI_COMPONENTS)
    
    return betti

# Parallelizing the feature extraction function to speed up the process
def preprocess_dataset(df):
    with Pool(N_THREADS) as p:
        X = p.map(extract_features, df['Path'])

    y = df['Emotions']

    return X, y

audio_paths = pd.read_csv(AUDIO_PATHS)
X, y = preprocess_dataset(audio_paths)

# Saving the preprocessed features (Betti curves) for future use
preprocessed = pd.DataFrame(X)
preprocessed['label'] = y
preprocessed.to_csv('features_gudhi.csv', index=False)