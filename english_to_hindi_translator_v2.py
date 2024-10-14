import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'path_to_your_dataset.csv' with the actual path)
dataset_path = './morpheme_pos_data.csv'
data = pd.read_csv(dataset_path)

# Inspect the dataset
print(data.head())

# Assuming the dataset has two columns: 'Word' and 'Morpheme_Type'
words = data['Word'].values
labels = data['POS'].values

# Encode the words and labels to numerical format
le_words = LabelEncoder()
le_labels = LabelEncoder()

X = le_words.fit_transform(words).reshape(-1, 1)
y = le_labels.fit_transform(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Define and train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1, c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

# Train CRF model
print(X_train)
print(y_train)
crf.fit(X_train, y_train)

# Predict using the CRF model
y_pred_crf = crf.predict(X_test)

# Evaluate the CRF model
print("CRF Classification Report:")
print(metrics.flat_classification_report(y_test, y_pred_crf, target_names=le_labels.classes_))

from sklearn import svm
from sklearn.metrics import classification_report

# Define and train SVM classifier
svm_classifier = svm.SVC()

svm_classifier.fit(X_train, y_train)

# Predict using the SVM model
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the SVM model
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=le_labels.classes_))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Convert labels to categorical format for LSTM training
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(le_labels.classes_))
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(le_labels.classes_))

# Define LSTM-based model
model = Sequential()
model.add(Embedding(input_dim=len(le_words.classes_), output_dim=64, input_length=1))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=len(le_labels.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM model
model.fit(X_train, y_train_cat, epochs=10, batch_size=16, validation_data=(X_test, y_test_cat))

# Predict and evaluate the LSTM model
y_pred_lstm = model.predict(X_test)
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)

print("LSTM Classification Report:")
print(classification_report(y_test, y_pred_lstm_classes, target_names=le_labels.classes_))


from transformers import BertTokenizer, TFBertForTokenClassification
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Load multilingual BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TFBertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(le_labels.classes_))

# Tokenize the input text
def tokenize_text(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

train_tokens = tokenize_text(words[:int(0.8*len(words))])
test_tokens = tokenize_text(words[int(0.8*len(words)):])

# Compile the BERT model
optimizer = Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss)

# Train the BERT model
model.fit(train_tokens['input_ids'], y_train_cat, epochs=3, batch_size=16)

# Predict using the BERT model
y_pred_bert = model.predict(test_tokens['input_ids'])
y_pred_bert_classes = np.argmax(y_pred_bert.logits, axis=-1)

print("mBERT Classification Report:")
print(accuracy_score(y_test, y_pred_bert_classes))
