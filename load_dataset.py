import os
import pandas as pd

RAVDESS_PATH = 'RAVDESS/'

# The function below conveniently sorts the datasets' files according the the audio files' emotions.
# The code is taken from: https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
# No significant changes to the original code
def load_ravdess(path):
    ravdess_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        
        actor = os.listdir(path + dir)
        
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            
            file_emotion.append(int(part[2]))
            file_path.append(path + dir + '/' + file)
            
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    
    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

    return Ravdess_df


audio_paths = load_ravdess(RAVDESS_PATH)
# Exporting the audio path-emotion pairs to a .csv
audio_paths.to_csv('audio_paths.csv', index=False)