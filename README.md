# Advanced Phishing Email Detection

This project aims to detect phishing emails using machine learning and deep learning techniques. It utilizes various models, including fine-tuning T5 for phishing email detection and traditional machine learning models like RandomForest, GradientBoosting, SVM, and MultinomialNB for classification.

## Project Structure

- **Fine-tuning T5 for Phishing Email Detection**: Fine-tuning of the T5 model for phishing email detection.
- **Traditional Machine Learning Models**: Implementation and evaluation of various traditional machine learning models.

## Installation

### Clone the Repository

```bash
git clone https://github.com/yashkadam435/Phishing-Email-Detection.git
cd Phishing-Email-Detection
```

## Install Required Libraries

```bash
pip install --upgrade transformers gradio==3.48.0 sentencepiece opendatasets pandas gdown matplotlib scikit-learn torch tqdm
```

## Dataset

Download the dataset from Kaggle using the following command:

```bash
gdown --id 1cOohptk4-83tBadQvdkjGE8AZ9Tp3yuW -O documents.zip
unzip -q documents.zip -d documents
```

Dataset Source: https://www.kaggle.com/datasets/subhajournal/phishingemails/

## Fine-tuning T5 for Phishing Email Detection

**Training Process**

1) Import the required libraries and dataset.
2) Preprocess the data by removing NaN values and unnecessary columns.
3) Split the data into training and testing sets.
4) Define a custom dataset class for T5.
5) Load the T5 tokenizer and model.
6) Train the model using the training data.
7) Evaluate the model using the testing data.
8) Save the fine-tuned model.

## Evaluation

Evaluate the fine-tuned model using accuracy and classification report metrics.

## Custom Input

Test the fine-tuned model for custom input using the provided function.

## Front-end using Gradio

Created a Gradio interface for the fine-tuned model.

## Traditional Machine Learning Models

**Data Preprocessing**

1) Import the required libraries and dataset.
2) Preprocess the data by removing NaN values and unnecessary columns.
3) Split the data into feature matrix X and target variable y.
4) Split the data into training and testing sets.
   
## Model Training and Evaluation

1) Train various models like RandomForest, GradientBoosting, SVM, and MultinomialNB.
2) Evaluate each model using accuracy, confusion matrix, and classification report.
3) Compare the accuracy of different models using a bar chart.
   
## Outputs

**Fine-tuning T5 for Phishing Email Detection**

1) Accuracy: 0.95
2) Classification Report: Precision, recall, and F1-score for each class.
   
**Traditional Machine Learning Models**

1) RandomForest Accuracy: 0.99
2) GradientBoosting Accuracy: 0.98
3) MultinomialNB Accuracy: 0.97
4) SVM Accuracy: 0.96

**Safe Email:**

![WhatsApp Image 2024-07-03 at 10 55 45 PM (1)](https://github.com/yashkadam435/Phishing-Email-Detection/assets/108817280/b470f857-8660-4036-af93-99e44152a28b)

**Phishing Email:**

![WhatsApp Image 2024-07-03 at 10 55 44 PM (1)](https://github.com/yashkadam435/Phishing-Email-Detection/assets/108817280/5d54957f-1d02-467e-89d1-3bb5131503c8)

## Usage

**Fine-tuning T5 for Phishing Email Detection**

1) Load the fine-tuned model and tokenizer.
2) Use the model for prediction on custom email text inputs.
   
**Traditional Machine Learning Models**

1) Train and evaluate various traditional machine learning models.
2) Compare the accuracy of different models using a bar chart.
   
## Contributing

Feel free to submit issues and enhancement requests.
