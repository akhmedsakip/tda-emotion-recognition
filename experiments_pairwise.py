import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

FEATURES_PATH = 'features_pairwise.csv'
NUM_SPLITS = 100

# Load the features/labels from a .csv file with the path specified in the function's argument
# The last column stands for the emotion label, so features are all the columns before that
def load_features(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df['label'].values
    return X, y

# Scale the features according to the passed argument specifying the scaling type (standard/minmax scaling, no scaling)
def scale_features(X, type):
    if type == 'standard':
        scaler = StandardScaler()
    elif type == 'minmax':
        scaler = MinMaxScaler()
    elif type == 'none':
        return X
    else:
        raise Exception("No valid scaler type passed.")
    return scaler.fit_transform(X)

# Perform dimensionality reduction pf the features via PCA based on the argument specifying n_components argument
# of scikit-learn's PCA
def reduce_features(X, n_components):
    if n_components == 'none':
        return X
    reducer = PCA(n_components=n_components)
    return reducer.fit_transform(X)

# Perform label encoding
def encode_labels(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y)

X_loaded, y_loaded = load_features(FEATURES_PATH)

approaches = {}

# Looping through all the approaches
for scaler in ['none', 'standard', 'minmax']:
    for reducer in ['none', 10, 0.80]:
            for classifier in ['rfc', 'knn', 'svc', 'mlp']:
                scores = []
                # Looping through all the train-test splits of data
                for i in range(NUM_SPLITS):
                    X, y = X_loaded.copy(), y_loaded.copy()
                    X = scale_features(X, type=scaler)
                    X = reduce_features(X, n_components=reducer)
                    y = encode_labels(y)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    # Running the respective algorithms of scikit-learn
                    if classifier == 'rfc':
                        clf = RandomForestClassifier(n_estimators=100)
                    elif classifier == 'knn':
                        clf = KNeighborsClassifier(n_neighbors=5)
                    elif classifier == 'svc':
                        clf = SVC(kernel='rbf')
                    elif classifier == 'mlp':
                        clf = MLPClassifier()
                    clf.fit(X_train, y_train)
                    score = f1_score(y_test, clf.predict(X_test), average='micro')
                    # Appending the f1-scores to the list of the splits' scores
                    scores.append(score)
                
                scores = np.array(scores)
                # Filling the dictionary with the mean and stdev of f1-scores for each split
                approaches[(scaler, reducer, classifier)] = (np.mean(scores), np.std(scores))

print(approaches)