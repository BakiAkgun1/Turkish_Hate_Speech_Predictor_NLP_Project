# Hate Speech Detection on Turkish Tweets

This repository implements **Detection of Hate Speech on Turkish Tweets Using Machine Learning** techniques. It includes preprocessing Turkish tweets, handling imbalanced datasets, text representation, synthetic data generation using TextGAN, and training multiple models to identify hate speech.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset Structure](#dataset-structure)
- [Preprocessing](#preprocessing)
- [Text Representation](#text-representation)
- [Models and Training](#models-and-training)
- [Addressing Class Imbalance](#addressing-class-imbalance)
- [TextGAN for Synthetic Data](#textgan-for-synthetic-data)
- [Performance Evaluation](#performance-evaluation)
- [Results and Discussion](#results-and-discussion)
- [References](#references)

---

## Introduction

The project focuses on detecting hate speech in Turkish tweets by experimenting with machine learning and deep learning approaches. It includes:
- **Data Preprocessing**: Cleaning, tokenizing, and lemmatizing text.
- **Class Imbalance Handling**: Using oversampling, undersampling, and synthetic data generation.
- **Model Evaluation**: Comparing multiple models, including boosting techniques and neural networks.

---

## Dataset Structure

The dataset is composed of Turkish tweets labeled as hate speech or non-hate speech. The following files were used:
- **`data.xlsx`**: The raw dataset.
- **`merged_data.csv`**: Cleaned and combined data from different sources.
- **`data_cleaned.csv`**: Final cleaned dataset with unnecessary columns removed.
- **`data_balanced.csv`**: Balanced dataset combining `data_cleaned.csv` and synthetic data generated via TextGAN.

### Synthetic Data
- **Synthetic tweets** were generated using TextGAN and added to balance the dataset, improving model training and evaluation.

---

## Preprocessing

### Steps:
1. **Cleaning**:
   - Removed URLs, special characters, and unnecessary symbols.
2. **Tokenization**:
   - Splitting tweets into individual words.
3. **Stopword Removal**:
   - Eliminated common meaningless words.
4. **Lemmatization**:
   - Converted words to their root forms for better feature representation.
![image](https://github.com/user-attachments/assets/1e215491-9d82-4cee-a708-dc62a8b1c48e)

---

## Text Representation

Two feature extraction methods were applied to convert tweets into numerical representations:
1. **CountVectorizer**:
   - Captures word frequencies using:
     - **Unigrams (n=1)** for single words.
     - **Bigrams (n=2)** for word pairs.
2. **TF-IDF Vectorizer**:
   - Represents word importance based on:
     - Term Frequency (TF): Occurrence in a document.
     - Inverse Document Frequency (IDF): Importance across all documents.
   - Both unigrams and bigrams were used for richer context representation.

---

## Models and Training

The following models were implemented and optimized:

### Machine Learning Models:
1. **CatBoost**:
   - Iterations: 20
   - Learning Rate: 0.05
   - Depth: 10
   - Random State: 42
2. **XGBoost**:
   - n_estimators: 100
   - Random State: 42
   - Achieved **best performance** with 82.09% accuracy.
3. **Random Forest**:
   - n_estimators: 100
   - Random State: 42
4. **Gradient Boosting**:
   - n_estimators: 100
   - Random State: 42

### Deep Learning Model:
1. **Artificial Neural Network (ANN)**:
   - Epochs: 5
   - Batch Size: 64
   - Used for capturing complex data relationships.
   - Monitored validation and training losses for performance.
![image](https://github.com/user-attachments/assets/5c9d797d-d69e-4c4a-a006-4ae07206aac4)

### Dimensionality Reduction:
- **TruncatedSVD**:
  - An efficient alternative to PCA for large and sparse datasets.
  - Reduced computation time while maintaining performance.

---

## Addressing Class Imbalance

The dataset suffered from class imbalance, which was addressed using the following techniques:
1. **Oversampling**:
   - Duplicated samples from the minority class to balance the dataset.
2. **Undersampling**:
   - Reduced the majority class samples for balance.
3. **Combined Resampling**:
   - Combined oversampling and undersampling for better results.

---

## TextGAN for Synthetic Data

To enhance the dataset, **TextGAN** was used for synthetic data generation:
- **Generator and Discriminator**:
  - Pre-trained models saved as `generator.pth` and `discriminator.pth`.
- **Synthetic Data Creation**:
  - Generated tweets were combined with the cleaned data (`data_cleaned.csv`) to produce a balanced dataset (`data_balanced.csv`).

This synthetic data significantly improved model training and generalization.

---

## Performance Evaluation

Performance was evaluated using the following metrics:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Percentage of correct positive predictions.
- **Recall**: Percentage of actual positives correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.

### Visualization:
1. **Confusion Matrices**:
   - Plotted to assess classification success and errors.
2. **Loss and Accuracy Trends**:
   - Tracked during training to identify overfitting or underfitting.

---

## Results and Discussion

### Best Model: **XGBoost**
- **Accuracy**: 82.09%
- **Precision**: 81.57%
- **Recall**: 82.09%
- **F1-Score**: 0.7928

### Model Comparisons:
![image](https://github.com/user-attachments/assets/364e4cd3-70ad-4a97-a956-f64133528a3d)

- **XGBoost**:
  - Outperformed all models in accuracy, precision, recall, and F1-score.
- **Gradient Boosting**:
  - Performed well but slightly lower than XGBoost.
- **Random Forest**:
  - Strong accuracy but weaker F1-score.
- **CatBoost**:
  - Faster but less precise.
- **ANN**:
  - Lowest accuracy, indicating that deep learning may require further optimization.

---

## References

1. TextGAN Paper: [https://arxiv.org/pdf/1708.07836.pdf](https://arxiv.org/pdf/1708.07836.pdf)  
2. Dataset 1:  https://github.com/imayda/turkish-hate-speech-dataset-2

---

This readme provides a comprehensive overview of the project. Let me know if additional details or changes are needed!
