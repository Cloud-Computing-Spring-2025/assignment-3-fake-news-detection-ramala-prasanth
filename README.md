# Fake News Detection using PySpark

This project implements a fake news detection pipeline using Apache PySpark. The workflow is divided into five modular tasks:

- **Task 1**: Data Loading & Preprocessing  
- **Task 2**: Text Cleaning  
- **Task 3**: Feature Extraction  
- **Task 4**: Model Training  
- **Task 5**: Model Evaluation

---

## 🧠 Task 1: Data Loading & Preprocessing

### Objective
Load the raw fake news dataset and apply initial preprocessing.

### Input
- `fake.csv`

### Output
- Cleaned DataFrame with selected columns: `id`, `title`, `text`, `label`

### Key Steps
- Load CSV using `spark.read.csv` with header and inferred schema
- Remove null or empty rows in required columns
- Select only relevant columns for downstream tasks

---

## 🧼 Task 2: Text Cleaning

### Objective
Preprocess and clean text for NLP by removing punctuation and stopwords.

### Input
- Preprocessed DataFrame from Task 1

### Output
- `task2_output/*.csv` containing `id`, `title`, `filtered_words`

### Key Steps
- Convert `text` to lowercase
- Remove punctuation and digits using regex
- Tokenize the text using `split` or a simple tokenizer
- Remove stopwords using a predefined list
- Output includes: `id`, `title`, `filtered_words` (list of cleaned tokens)

---

## 📊 Task 3: Feature Extraction

### Objective
Transform the cleaned text data into TF-IDF vectors for modeling.

### Input
- Output from Task 2

### Output
- `task3_output/*.csv` with: `id`, `filtered_words`, `features`, `label_index`

### Key Steps
- Use `CountVectorizer` to convert tokens into term frequency vectors
- Apply `IDF` for weighting features
- Use `StringIndexer` to encode labels (e.g., fake → 0, real → 1)
- Output `features` as a stringified sparse vector and `label_index` for modeling

---

## 🤖 Task 4: Model Training

### Objective
Train a logistic regression classifier using TF-IDF features.

### Input
- Parsed features and labels from Task 3
- `title` from Task 2 (joined by `id` for reporting)

### Output
- `task4_output/*.csv` with: `id`, `title`, `label_index`, `prediction`

### Key Steps
- Convert stringified sparse vectors back into `SparseVector` using UDF
- Train/test split (e.g., 80/20)
- Train `LogisticRegression` model on training set
- Apply model to test set
- Join with original `title` for output
- Write predictions as CSV with selected fields

---

## 📈 Task 5: Model Evaluation

### Objective
Evaluate the model using standard classification metrics.

### Input
- Predictions from Task 4

### Output
- `task5_output.csv` or printed markdown table with evaluation results

### Key Steps
- Use `MulticlassClassificationEvaluator` to compute:
  - Accuracy
  - F1 Score
- Save metrics to `task5_output.csv` or display them as:

```markdown
| Metric   | Value |
|----------|-------|
| Accuracy | 0.89  |
| F1 Score | 0.88  |
| Precision | 0.87  |
| Recall    | 0.89  |
```

## 💻 Requirements

- Python 3.12+
- PySpark
- Standard Python libraries: `re`, `os`, etc.

### Install requirements (if needed):

```bash
pip install pyspark
```

## 🚀 How to Run
```
python Task1/data_loading.py
python Task2/text_cleaning.py
python Task3/feature_extraction.py
python Task4/model_training.py
python Task5/evaluation.py

```
## 📁 Directory Structure
```
.
├── Task1/
│   └── data_loading.py
├── Task2/
│   └── text_cleaning.py
├── Task3/
│   └── feature_extraction.py
├── Task4/
│   └── model_training.py
├── Task5/
│   └── evaluation.py
├── fake.csv
├── README.md
└── requirements.txt
```


