# BERT Sentiment Analysis using Hugging Face

This repository contains an end-to-end implementation of sentiment analysis using a pre-trained BERT model fine-tuned with the Hugging Face Transformers library. The project demonstrates a complete NLP workflow, starting from exploratory data analysis to model training, evaluation, and saving the trained model for reuse.

---

## Project Overview

The goal of this project is to build a robust sentiment classification model using BERT. The notebook follows best practices for modern NLP pipelines, including stratified data splitting, Hugging Face Dataset integration, metric-based evaluation, and efficient model fine-tuning.

---

## Dataset

- The dataset is sourced from **GitHub**
- It contains labeled text data for sentiment classification
- The data is split into **train, validation, and test sets** using stratified sampling to preserve class distribution

---

## Workflow

The project follows the steps below:

1. **Library Setup**
   - PyTorch
   - Hugging Face Transformers & Datasets
   - Scikit-learn, evaluate
   - NumPy, Pandas, matplotlib, seaborn

2. **Exploratory Data Analysis (EDA)**
   - Basic inspection of text samples
   - Label distribution analysis

3. **Stratified Train–Validation–Test Split**
   - Ensures balanced class representation across splits

4. **Hugging Face Dataset Conversion**
   - Conversion from Pandas DataFrame to `Dataset` and `DatasetDict`

5. **Label Encoding**
   - Creation of `label2id` and `id2label` mappings for model compatibility

6. **Model Selection**
   - Pre-trained **BERT** model for sequence classification
   - Loaded using Hugging Face Transformers

7. **Tokenization**
   - Tokenization using BERT tokenizer
   - Truncation and padding applied
   - Removal of unnecessary columns to reduce memory usage

8. **Training Configuration**
   - TrainingArguments configured for:
     - Evaluation during training
     - Logging
     - Model checkpointing

9. **Evaluation Metrics**
   - Accuracy
   - F1-score
   - Precision

10. **Model Training**
    - Fine-tuning performed using Hugging Face `Trainer` API

11. **Prediction and Evaluation**
    - Model evaluated on the test dataset
    - Metrics reported using standard NLP evaluation practices

12. **Model Saving**
    - Fine-tuned model and tokenizer saved for future inference or deployment

---

## Results

The fine-tuned BERT model achieved the following performance on the test set:

- **Accuracy:** 93.18%
- **F1 Score:** 93.19%
- **Test Loss:** 0.1966
- **Evaluation Runtime:** 15.95 seconds
- **Samples per Second:** 200.59

These results demonstrate strong generalization and effective fine-tuning of the BERT model for sentiment classification.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Google Colab (GPU)

---

