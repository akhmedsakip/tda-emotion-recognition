# Topological Data Analysis for Speech Emotion Recognition
## Course Project
## By Akhmed Sakip, Noor Hussein, Nurdaulet Mukhituly
Reach out at akhmedsakip@gmail.com

### Downloading & extracting the dataset
The project uses the RAVDESS dataset. To download it, head [here](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) and download the dataset. Create a folder named 'RAVDESS' and extract the archive's contents there. 
<br>The resulting data paths should look like 'RAVDESS / Actor_** / *.wav'.

### Downloading additional data before running the project
To avoid generating persistence images from scratch in *generate_persims.py*, download the contents (<2 GB) of [this folder](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/akhmed_sakip_mbzuai_ac_ae/EkeSvj0lkdlBtMMjDfXpTMYBbzayUmbmsvegHDSmc1vuLQ?e=bdgJhJ) and place the 2 files (*features_ts2pi.txt*, *labels_ts2pi.csv*) directly to the project's folder (alongside with other project files).

### Packages to install
The project was tested on **Python 3.10.4**, so to be on the safe side, it is preferable to use one of the releases of **Python 3.10**.
<br>Once you have created an environment running Python 3.10, make sure to install the following packages using *pip*:
<br>
<br>**NumPy**:
<br>&emsp;&emsp;&emsp; `pip install numpy`
<br>
<br>**pandas**:
<br>&emsp;&emsp;&emsp; `pip install pandas`
<br>
<br>**scikit-learn**:
<br>&emsp;&emsp;&emsp; `pip install scikit-learn`
<br>
<br>**librosa**:
<br>&emsp;&emsp;&emsp; `pip install librosa`
<br>
<br>**gudhi**:
<br>&emsp;&emsp;&emsp; `pip install gudhi`
<br>
<br>**giotto-tda**:
<br>&emsp;&emsp;&emsp; `pip install giotto-tda`
<br>
<br>**tqdm**:
<br>&emsp;&emsp;&emsp; `pip install tqdm`
<br>
<br>**tensorflow**:
<br>&emsp;&emsp;&emsp; `pip install tensorflow`
<br>
<br>*Note*: if you need GPU acceleration, follow the instructions [here](https://www.tensorflow.org/install/pip). Otherwise, you had better run the TensorFlow code (*vgg16_ts2pi.py*) on Kaggle, since setting up GPU acceleration on the university's lab computers is complicated and problematic.
<br>

*Warning*: **ts2pi** cannot be installed alongside the newest versions of the above packages and is thus incompatible with them. If you wish to regenerate persistence images without downloading them from the link provided above, create a new environment and install ONLY the package below, so that it will pull the specific versions by itself.
<br>**ts2pi**:
<br>&emsp;&emsp;&emsp; `pip install ts2pi`
<br>

### Repository contents

The project directory consists of the following Python files:
<br>
<br>&emsp;&emsp;**load_dataset.py** - conveniently formats the audio paths and sorts them according to emotions, saving the results to *audio_paths.csv*. The code for processing the dataset folder is taken from [here](https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition).
<br>
<br>&emsp;&emsp;**experiments_betti.py** - runs classification experiments on Betti curves generated using the *gudhi* package.
<br>
<br>&emsp;&emsp;**experiments_giottobetti.py** - runs classification experiments on Betti curves generated using manual preprocessing and the *giotto-tda* package.
<br>
<br>&emsp;&emsp;**experiments_landscapes.py** - runs classification experiments on persistence landscapes generated using manual preprocessing and the *giotto-tda* package.
<br>
<br>&emsp;&emsp;**experiments_pairwise.py** - runs classification experiments on pairwise distances between persistence diagrams generated using manual preprocessing and the *giotto-tda* package.
<br>
<br>&emsp;&emsp;**vgg16_ts2pi.py** - runs classification experiments by training the VGG16 CNN with persistence images generated using the *ts2pi* package.
<br>
<br>&emsp;&emsp;**generate_betticurves.py** - generates Betti curves for the audio files using the *gudhi* package by following the approach outlined [here](https://ieeexplore.ieee.org/document/9356319). The results are stored in *features_gudhi.csv*.
<br>
<br>&emsp;&emsp;**generate_giotto.py** - generates Betti curves and persistence landscapes for the audio files using manual preprocessing and the *giotto-tda* package. The results are stored in *features_giottobetti.csv* and *features_landscapes.csv*.
<br>
<br>&emsp;&emsp;**generate_pairwise.py** - calculates pairwise distances for the persistence diagrams of the audio files using manual preprocessing and the *giotto-tda* package. The results are stored in *features_pairwise.csv*.
<br>
<br>&emsp;&emsp;**generate_persims.py** - generates persistence images for the audio files using the *ts2pi* package. The results are stored in *features_ts2pi.txt* and *labels_ts2pi.csv*. To download them, refer above.

### Replication of experiments
To replicate the classification experiments, simply run the corresponding Python files named *experiments_\*.py* and *vgg16_ts2pi.py* inside an environment with installed packages. All the sample preprocessed data is already provided in this repository (except for VGG16, which should be downloaded referring to the above instructions), so you do not have to rerun the *generate_\*.py* files.
<br><br>
**Warning**: the Python files are set to run experiments for 100 train-test splits, which could take considerable time. To reduce the number of splits, simply change the *NUM_SPLITS* in the files' beginnings to a smaller number.
<br><br>
In case you wish to run the preprocessing files, first run *load_dataset.py* and proceed to running the *generate_\*.py* files.

### References
[1] M. Tlachac, A. Sargent, E. Toto, R. Paffenroth and E. Rundensteiner, "Topological Data Analysis to Engineer Features from Audio Signals for Depression Detection," _2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA)_, 2020, pp. 302-307, doi: 10.1109/ICMLA51294.2020.00056.
<br>
[2] The GUDHI Project, GUDHI User and Reference Manual. GUDHI Editorial Board, 2015. [Online]. Available: http://gudhi.gforge.inria.fr/doc/latest/
<br>
[3] G. Tauzin et al., giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration. 2020. [Online]. Available: https://giotto-ai.github.io/gtda-docs/0.5.1/library.html
<br>
[4] K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition.” arXiv, 2014. doi: 10.48550/ARXIV.1409.1556.
