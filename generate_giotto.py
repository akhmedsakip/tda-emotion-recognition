import numpy as np
import pandas as pd
import librosa as lr
import gudhi
from multiprocessing import Pool
import math
from gtda.diagrams import BettiCurve, PersistenceLandscape

AUDIO_PATHS = 'audio_paths.csv'
# Modify the below variable depending on the number of CPU threads you are ready to allocate
N_THREADS = 16


# Extracting persistence barcode given the path of an audio file
# This code is significantly changed based on the original code from https://github.com/mltlachac/TDA
def extract_persistence_barcode(path):
    data, sr = lr.load(path, sr=16000, duration=2.5, offset=0.6)
    simplex = gudhi.SimplexTree()
    for i in np.arange(len(data)):
        simplex.insert([i, i+1], filtration=data[i])
    barcode = simplex.persistence()
    return barcode


# Function for postprocessing the diagrams
# from the source code of giotto-tda at https://github.com/giotto-ai/giotto-tda/blob/master/gtda/homology/_utils.py
def _postprocess_diagrams(
        Xt, format, homology_dimensions, infinity_values, reduced
        ):
    
    def replace_infinity_values(subdiagram):
        np.nan_to_num(subdiagram, posinf=infinity_values, copy=False)
        return subdiagram[subdiagram[:, 0] < subdiagram[:, 1]]

    
    if format in ["ripser", "flagser"]:  
        
        slices = {dim: slice(None) if (dim or not reduced) else slice(None, -1)
                  for dim in homology_dimensions}
        Xt = [{dim: replace_infinity_values(diagram[dim][slices[dim]])
               for dim in homology_dimensions} for diagram in Xt]
    elif format == "gudhi":  
        
        slices = {dim: slice(None) if (dim or not reduced) else slice(1, None)
                  for dim in homology_dimensions}
        Xt = [{dim: replace_infinity_values(
            np.array([pers_info[1] for pers_info in diagram
                      if pers_info[0] == dim]).reshape(-1, 2)[slices[dim]]
            )
            for dim in homology_dimensions} for diagram in Xt]
    else:
        raise ValueError(
            f"Unknown input format {format} for collection of diagrams."
            )

    
    start_idx_per_dim = np.cumsum(
            [0] + [np.max([len(diagram[dim]) for diagram in Xt] + [1])
                   for dim in homology_dimensions]
            )
    
    # Changed min() to np.min() and .size to len()
    min_values = [np.min([np.min(diagram[dim][:, 0]) if len(diagram[dim])
                       else np.inf for diagram in Xt])
                  for dim in homology_dimensions]
    min_values = [min_value if min_value != np.inf else 0
                  for min_value in min_values]
    n_features = start_idx_per_dim[-1]
    Xt_padded = np.empty((len(Xt), n_features, 3), dtype=float)

    for i, dim in enumerate(homology_dimensions):
        start_idx, end_idx = start_idx_per_dim[i:i + 2]
        padding_value = min_values[i]
        
        Xt_padded[:, start_idx:end_idx, 2] = dim
        for j, diagram in enumerate(Xt):
            subdiagram = diagram[dim]
            end_idx_nontrivial = start_idx + len(subdiagram)
            
            Xt_padded[j, start_idx:end_idx_nontrivial, :2] = subdiagram
         
            Xt_padded[j, end_idx_nontrivial:end_idx, :2] = [padding_value] * 2

    return Xt_padded

# Parallelizing the feature extraction function to speed up the process
def preprocess_dataset(df):
    with Pool(N_THREADS) as p:
        X = p.map(extract_persistence_barcode, df['Path'])

    y = df['Emotions']

    X = _postprocess_diagrams(
            X, "gudhi", sorted([0, 1]), np.inf, True) 

    return X, y


audio_paths = pd.read_csv(AUDIO_PATHS)
X, y = preprocess_dataset(audio_paths)

# Extracting Betti curves using giotto-tda
bc = BettiCurve()
X_bc = bc.fit_transform(X)
X_bc = X_bc.reshape(X_bc.shape[0], X_bc.shape[1] * X_bc.shape[2])

# Saving the preprocessed features (Betti curves) for future use
preprocessed = pd.DataFrame(X_bc)
preprocessed['label'] = y
preprocessed.to_csv('features_giottobetti.csv', index=False)

# Extracting persistence landscapes using giotto-tda
pc = PersistenceLandscape()
X_pc = pc.fit_transform(X)
X_pc = X_pc.reshape(X_pc.shape[0], X_pc.shape[1] * X_pc.shape[2])

# Saving the preprocessed features (persistence landscapes) for future use
preprocessed = pd.DataFrame(X_pc)
preprocessed['label'] = y
preprocessed.to_csv('features_landscapes.csv', index=False)