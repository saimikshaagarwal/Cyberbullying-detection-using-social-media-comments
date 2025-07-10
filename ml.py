import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("C:/Users/Lenovo/Downloads/hateXplain.csv")
print("Dataset shape:", df.shape)
print(df.columns)
print(df.iloc[0])

# Clean text
text_column = 'post_tokens'
label_column = 'label'
df[text_column] = df[text_column].astype(str)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

df['clean_text'] = df[text_column].apply(clean_text)
df = df.dropna()
print(df['clean_text'])

# Map to binary classes: 1 = cyberbullying, 0 = non-cyberbullying
label_mapping = {'hatespeech': 1, 'offensive': 1, 'normal': 0}
df['binary_label'] = df[label_column].map(label_mapping)

# Class distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x='binary_label', data=df)
plt.title('Class Distribution: Cyberbullying vs Non-Cyberbullying')
plt.xlabel('Label (0 = Non-Cyberbullying, 1 = Cyberbullying)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Comment length distribution
df['text_length'] = df['post_tokens'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8, 4))
sns.histplot(df['text_length'], bins=30, kde=True, color='orange')
plt.title("Comment Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['clean_text'])
y_binary = df['binary_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_binary, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression (Binary): {:.2f}%".format(accuracy * 100))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Non-Cyberbullying', 'Cyberbullying']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Cyberbullying Detection')
plt.tight_layout()
plt.show()

# Classification Report
cr = classification_report(y_test, y_pred, target_names=labels)
print("Classification Report:\n", cr)