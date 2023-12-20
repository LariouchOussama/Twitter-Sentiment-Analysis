# Twitter Sentiment Analysis

This repository contains a Jupyter notebook for performing sentiment analysis on Twitter data using state-of-the-art natural language processing models, including BERT, LSTM, and GPT. The notebook covers the entire pipeline, from data preprocessing and exploration to model training, evaluation, and visualization of results.

## Overview

- **Data Preprocessing:** The notebook preprocesses Twitter data by cleaning and transforming text, removing noise, and preparing it for analysis.

- **Exploratory Data Analysis (EDA):** Visualizations are provided to understand the distribution of sentiments in the dataset.

- **Modeling:** Three powerful models—BERT, LSTM, and GPT—are implemented for sentiment classification. The notebook utilizes the Hugging Face Transformers library for BERT and GPT.

- **Training and Evaluation:** The models are trained on the dataset, and their performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

- **Confusion Matrix:** Visual representations of confusion matrices are included for a detailed understanding of model performance.

## Prerequisites

- Python
- Jupyter Notebook
- TensorFlow
- Hugging Face Transformers
- Plotly
- NLTK
- Pandas
- NumPy

# Twitter Sentiment Analysis using LSTM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V5a738HNZQjy120htn1p8qYoRse-OdOL?usp=sharing)

This notebook explores sentiment analysis on Twitter using the LSTM (Long Short-Term Memory) architecture. The goal is to classify tweets into positive or negative sentiment categories.

## Dataset

The dataset used in this project is sourced from Kaggle: [training.1600000.processed.noemoticon.csv](https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv/data)

## Overview

The project involves the following key steps:

1. **Data Preprocessing:** Cleaning and preparing the dataset for sentiment analysis.

2. **Text Processing:** Tokenization, sequence padding, and other text processing steps.

3. **Model Architecture:** Implementing LSTM (Long Short-Term Memory) model for sentiment analysis.

4. **Training and Evaluation:** Training the model and evaluating its performance on a test set.

## How to Use

To run the notebook:

1. Click on the "Open in Colab" badge above or use the following link: [Open In Colab](https://colab.research.google.com/drive/1V5a738HNZQjy120htn1p8qYoRse-OdOL?usp=sharing).
2. Follow the instructions in the notebook cells to execute each step.

## Results

The trained model achieved the following performance on the test set:


          precision    recall  f1-score   support
       0       0.74      0.67      0.70      1012
       1       0.69      0.77      0.73       988
       accuracy                           0.72      2000



# Twitter Sentiment Analysis using BERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11sfKeJ_hFtpw1DnJi4YgYjsU9eOFmoL-?usp=sharing)

This notebook explores sentiment analysis on Twitter using the BERT (Bidirectional Encoder Representations from Transformers) architecture. The goal is to classify tweets into positive or negative sentiment categories.

## Dataset

The dataset used in this project is sourced from Kaggle: [training.1600000.processed.noemoticon.csv](https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv/data)

## Overview

The project involves the following key steps:

1. **Data Preprocessing:** Cleaning and preparing the dataset for sentiment analysis.

2. **Text Processing:** Tokenization and sequence padding of text data.

3. **BERT Model:** Utilizing the powerful BERT model for sentiment classification.

4. **Training and Evaluation:** Training the model and evaluating its performance on a test set.

## How to Use

To run the notebook:

1. Click on the "Open in Colab" badge above or use the following link: [Open In Colab](https://colab.research.google.com/drive/11sfKeJ_hFtpw1DnJi4YgYjsU9eOFmoL-?usp=sharing).
2. Follow the instructions in the notebook cells to execute each step.

## Results

The trained BERT model achieved the following performance on the test set:

          precision    recall  f1-score   support
       0       0.75      0.79      0.77      1012
       1       0.77      0.73      0.75       988
       accuracy                           0.76      2000


## Confusion Matrix

Visual representation of model predictions versus true labels:
![Confusion Matrix](https://github.com/LariouchOussama/Twitter-Sentiment-Analysis/assets/90968210/a98f7a24-7efa-481e-8e32-4db1d7af5f0a)


Feel free to explore the notebook, experiment with different configurations, and contribute to the project!

<!-- Add any additional sections or details about your project as needed -->



Feel free to explore, modify, and experiment with the notebook to adapt it to your specific use case.
