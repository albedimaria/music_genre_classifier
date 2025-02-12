# Music Genre Classifier
This repository contains a deep learning-based music genre classification system implemented with Neural Networks (NN), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN + LSTM). The models are trained on MFCC-extracted audio features to predict the genre of music tracks.

## Features
Three deep learning-based architectures:

- NN (Standard Fully Connected Network)

- CNN (Convolutional Neural Network)

- RNN + LSTM (Recurrent Neural Network + Long Short Term Memory)

Other features:
- MFCC Feature Extraction from raw audio using librosa
- Performance Evaluation with accuracy and loss plots
- Scalable and Extendable, supporting new datasets and architectures.

## Installation and Setup
1. Clone the repository 
```
git clone https://github.com/albedimaria/music_genre_classifier.git
cd music_genre_classifier
```
2. Create a Virtual Environment (Optional)
```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
3. Install Dependencies
```
pip install -r requirements.txt
```

## Dataset
- The project originally used the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset
- MFCCs have to be extracted and will be stored in data.json.

## Future Improvements

- Use Spectrograms instead of MFCCs to improve feature extraction.
- Apply Transfer Learning using pre-trained audio models like VGGish or OpenL3.
- Optimize Hyperparameters using grid search or Bayesian optimization.
