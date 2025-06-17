# NLP-based-Resume-Classifier

An end-to-end machine learning pipeline to classify resumes from unstructured `.pdf` / `.docx` files and structured `.csv` metadata. This project was developed as part of the **Piramal Finance Hackathon 2024**, combining advanced NLP techniques and classical machine learning models.

---

## 🚀 Pipeline Overview

### 📄 Document Parsing
- Used `PlumPDF`, `PyPDF2`, and `python-docx` to extract raw text from PDF and DOCX resumes.

### 🧹 Preprocessing
- Utilized `regex` and `NLTK` for:
  - Tokenization  
  - Stopword removal  
  - Text normalization

### 🧠 Feature Engineering
- Text features:
  - `TF-IDF`  
  - `Word2Vec`  
  - Optional: LLM-based embeddings (e.g. OpenAI/BERT)
- Metadata features:
  - Encoded using `LabelEncoder` from structured `.csv` data

### 📊 Modeling
- Trained models:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Libraries: `scikit-learn`, `xgboost`

### 📈 Scoring & Evaluation
- Metrics:
  - Accuracy  
  - F1 Score  
  - ROC-AUC  
- Resume Scoring:
  - Based on skill and keyword match with job requirements

---

## 🧾 Technologies Used

| Component            | Stack                                      |
|----------------------|--------------------------------------------|
| Parsing              | PlumPDF, PyPDF2, python-docx               |
| NLP Preprocessing    | NLTK, regex, pandas, NumPy                 |
| Feature Extraction   | TF-IDF, Word2Vec, LLM embeddings (optional)|
| Modeling             | scikit-learn, XGBoost                      |
| Evaluation           | sklearn.metrics, seaborn, matplotlib       |

---

## 📌 Future Improvements
- Integrate full support for LLM-based embeddings (BERT, OpenAI, etc.)
- Deploy the pipeline as a web-based resume screening tool
- Include feedback-based model fine-tuning loop

---

## 📣 Acknowledgments
This project was built for the **Piramal Finance Hackathon 2024**.  
We thank the organizers and mentors for their guidance and support.
