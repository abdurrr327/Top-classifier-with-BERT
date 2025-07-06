# News Topic Classifier with BERT
This project demonstrates how to fine-tune a pre-trained BERT model for a multi-class text classification task. The goal is to accurately classify news headlines into one of four categories: World, Sports, Business, or Sci/Tech.
## Objective
The main objective is to build and evaluate a high-performing text classifier using the Hugging Face transformers library. This involves:
Loading and preprocessing the AG News dataset.
Tokenizing the text data to be compatible with bert-base-uncased.
Fine-tuning the pre-trained BERT model on the classification task.
Evaluating the model's performance using accuracy and F1-score.
Creating a simple, interactive web application using Gradio and Streamlit to demonstrate the model's capabilities on new, unseen headlines.
## Dataset
The project utilizes the AG News dataset, which is a standard benchmark for text classification. It is readily available from the Hugging Face datasets library.
Content: A collection of news articles, each with a headline (text) and a corresponding category label (label).
## Classes:
0: World
1: Sports
2: Business
3: Sci/Tech
Splits: The dataset is pre-divided into train and test sets, which we use for training and final evaluation, respectively.
## Methodology / Approach
The process is broken down into several key stages, all implemented within the provided Google Colab notebook.
1. Data Loading and Preprocessing
The AG News dataset is loaded directly from the Hugging Face Hub using the datasets library.
We inspect the dataset's structure and define mappings from integer labels to human-readable names (e.g., 0 -> World).
To ensure efficient training and evaluation, we create smaller, stratified subsets of the training and test data. This is particularly useful for demonstrations and running in resource-constrained environments like Google Colab.
2. Model and Tokenizer
We selected bert-base-uncased as our base model. It's a powerful and widely-used transformer model that provides a strong foundation for fine-tuning.
The corresponding tokenizer is loaded to convert raw text headlines into the numerical format (input IDs, attention masks) that BERT understands.
A tokenization function is mapped across the entire dataset, processing the text in batches for efficiency.
3. Fine-Tuning
The fine-tuning process is managed by the Hugging Face Trainer API, which simplifies the training loop.
We use the AutoModelForSequenceClassification class, which adds a classification head on top of the pre-trained BERT model. This head is initialized with random weights and is the primary part of the model that gets trained.
Training Arguments are defined, including:
num_train_epochs: 3
learning_rate: 2e-5
per_device_train_batch_size: 16
weight_decay: 0.01
The model is trained on the preprocessed training set and evaluated on a validation set at the end of each epoch to monitor progress and prevent overfitting.
4. Evaluation
The performance of the fine-tuned model is measured on the unseen test set.
Metrics:
Accuracy: The overall percentage of correctly classified headlines.
F1-Score (Macro): The harmonic mean of precision and recall, calculated for each class and then averaged. This is a robust metric for multi-class classification, especially if class imbalance were an issue.
A Confusion Matrix is generated and visualized using matplotlib and seaborn to provide a detailed, per-class view of the model's performance, showing where it makes correct and incorrect predictions.
5. Deployment
The fine-tuned model is saved to disk.
Two simple interactive UIs are created to demonstrate the model:
Gradio: A simple interface is built directly within the Colab notebook for quick, interactive testing.
Streamlit: A standalone Python script (app.py) is provided. This script can be run locally to launch a more polished web application. Instructions for running it are included in the notebook.
## Key Results & Observations
The fine-tuned model performed exceptionally well on the test set.
Metric	Score
Test Accuracy	92.56%
Test F1-Score (Macro)	0.92
## Confusion Matrix
(You can add the generated confusion matrix image here in your GitHub README)
![alt text](confusion_matrix.png)
The model achieved an accuracy of ~92.6%, which is excellent for this task and demonstrates the power of fine-tuning pre-trained models.
The confusion matrix shows high values along the diagonal, indicating that the model correctly classified the vast majority of headlines for each category.
The most frequent confusion was between World and Business news, which is understandable as these topics often overlap (e.g., global economic news).
The Sports and Sci/Tech categories were classified with very high precision and recall, showing the model's ability to learn the distinct vocabulary of these domains.
## How to Run
Colab Notebook:
Open the News_Topic_Classifier.ipynb file in Google Colab.
Ensure the runtime is set to use a GPU (Runtime -> Change runtime type -> T4 GPU).
Run the cells sequentially from top to bottom.
The model will be trained, evaluated, and a Gradio demo will launch at the end.
## Streamlit App (Locally):
After running the Colab notebook, a bert-ag-news-classifier.zip file will be created. Download it.
Unzip the file in your local project directory.
Download the app.py file.
Ensure you have streamlit installed: pip install streamlit.
Run the app from your terminal: streamlit run app.py.
