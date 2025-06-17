# NLP-based-Resume-Classifier

Document Parsing
PlumPDF, PyPDF2, python-docx used to extract raw text.
Preprocessing
regex, NLTK: tokenization, stopword removal, text normalization.
Feature Engineering
TF-IDF, Word2Vec, and optional LLM embeddings (e.g. OpenAI/BERT).
Metadata features from CSV encoded via LabelEncoder.
Modeling
Trained Logistic Regression, Random Forest, and XGBoost using scikit-learn and xgboost.
Scoring & Evaluation
Accuracy, F1, ROC-AUC metrics.
Resume scoring based on skill/keyword match


🧾 Technologies Used
Component	Stack
Parsing	PlumPDF, PyPDF2, python-docx
NLP Preprocessing	NLTK, regex, pandas, NumPy
Feature Extraction	TF-IDF, Word2Vec, LLM embeddings (optional)
Modeling	scikit-learn, XGBoost
Evaluation	sklearn.metrics, seaborn, matplotlib
