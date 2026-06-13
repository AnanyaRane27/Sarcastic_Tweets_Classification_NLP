Hindi Sarcasm Detection in Tweets
This project builds an end-to-end NLP pipeline to detect sarcasm in Hindi tweets using classical machine learning. The dataset contains 16,179 Hindi tweets (6,051 sarcastic + 10,128 non-sarcastic).
The core challenge is processing Devanagari script and code-mixed Hinglish text — standard NLP tools fail on Hindi, so a custom preprocessing pipeline was built using the indic-nlp-library for tokenization, along with bilingual (Hindi + English) stopword removal, emoji stripping, and TF-IDF vectorization with bigrams.
Three classifiers were trained and evaluated — Naive Bayes, Gradient Boosting, and XGBoost — alongside unsupervised clustering (K-Means, Agglomerative, GMM) and LDA topic modeling. Models were evaluated on Accuracy, F1-Score, ROC-AUC, and Confusion Matrix. BERT/mBERT fine-tuning was planned as an extension but remains as future work.
Tech Stack: Python, Scikit-learn, XGBoost, indic-nlp-library,Google Colab
