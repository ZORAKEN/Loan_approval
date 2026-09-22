# Loan_approval
# 🏦 Loan Approval Prediction using Machine Learning

A machine learning project that predicts whether a loan application is likely to be **approved or rejected** based on applicant and loan-related information.

The project demonstrates a complete machine-learning workflow, including **data preprocessing, exploratory data analysis, categorical encoding, train-test splitting, Support Vector Machine classification, and model evaluation** using Python and Scikit-learn.

---

## 📌 Project Overview

Loan approval is influenced by several factors such as:

* Applicant's gender
* Marital status
* Number of dependents
* Education
* Employment status
* Applicant income
* Co-applicant income
* Loan amount
* Loan term
* Credit history
* Property area

This project uses these features to build a classification model that predicts the **Loan Status** of an applicant.

### 🎯 Objective

> Build a machine learning classification model capable of predicting loan approval based on historical loan application data.

---

## 🛠️ Technologies Used

| Technology          | Purpose                             |
| ------------------- | ----------------------------------- |
| 🐍 Python           | Programming language                |
| 🐼 Pandas           | Data manipulation and preprocessing |
| 🔢 NumPy            | Numerical operations                |
| 📊 Seaborn          | Data visualization                  |
| 🤖 Scikit-learn     | Machine learning                    |
| 📓 Jupyter Notebook | Development environment             |

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Missing Value Handling
   ↓
Categorical Feature Encoding
   ↓
Exploratory Data Analysis
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Support Vector Machine (SVM)
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Loan Approval Prediction
```

---

## 📂 Dataset

The project uses a loan application dataset containing applicant information and a target variable:

**Target:** `Loan_Status`

* `1` → Loan Approved
* `0` → Loan Rejected

The dataset is loaded using Pandas:

```python
loan_dataset = pd.read_csv('dataset.csv')
```

---

## 🧹 Data Preprocessing

Several preprocessing steps were performed before training the model.

### Handling Missing Values

Missing observations were identified and removed:

```python
loan_dataset.isnull().sum()

loan_dataset = loan_dataset.dropna()
```

### Encoding the Target Variable

The categorical loan status was converted into numerical values:

```python
loan_dataset.replace(
    {"Loan_Status": {"N": 0, "Y": 1}},
    inplace=True
)
```

### Encoding Categorical Features

Categorical variables were converted into numerical representations.

For example:

```text
Gender
Male   → 1
Female → 0
```

```text
Married
Yes → 1
No  → 0
```

```text
Education
Graduate     → 1
Not Graduate → 0
```

```text
Property Area
Rural     → 0
Semiurban → 1
Urban     → 2
```

The `Dependents` feature was also normalized by replacing `3+` with `4`.

---

## 📊 Exploratory Data Analysis

The project includes visual analysis of the relationship between applicant characteristics and loan approval.

Examples include:

### Education vs Loan Status

A count plot was used to examine the relationship between education level and loan approval.

### Marital Status vs Loan Status

A count plot was also used to analyze loan approval across different marital-status groups.

These visualizations provide an initial understanding of patterns within the dataset.

---

## 🧮 Feature Selection

The `Loan_ID` column was removed because it is an identifier rather than a predictive feature.

The dataset was separated into features `X` and target `Y`:

```python
X = loan_dataset.drop(
    columns=['Loan_ID', 'Loan_Status'],
    axis=1
)

Y = loan_dataset['Loan_Status']
```

---

## ✂️ Train-Test Split

The dataset was divided into training and testing sets using Scikit-learn:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.1,
    stratify=Y,
    random_state=2
)
```

The split uses **stratification** to preserve the class distribution between the training and testing sets.

---

## 🤖 Machine Learning Model

### Support Vector Machine (SVM)

A Support Vector Machine classifier with a linear kernel was selected:

```python
classifier = svm.SVC(kernel='linear')
```

The model was then trained using the training dataset:

```python
classifier.fit(X_train, Y_train)
```

SVM attempts to find a decision boundary that separates the two classes:

```text
              Loan Approved
                   ↑
             ● ● ● ● ●
           ● ● ● ●
        ------------------  ← Decision Boundary
       ○ ○ ○ ○
     ○ ○ ○ ○ ○
                   ↓
              Loan Rejected
```

---

## 📈 Model Evaluation

The trained model is evaluated on both the training and test datasets using **classification accuracy**.

### Training Accuracy

```python
X_train_prediction = classifier.predict(X_train)

training_data_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)
```

### Test Accuracy

```python
X_test_prediction = classifier.predict(X_test)

test_data_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)
```

The notebook prints both values to assess how well the SVM performs on the available data.

> **Note:** The README intentionally does not claim a specific accuracy because the notebook source does not provide the actual printed output values.

---

## 🧠 Key Machine Learning Concepts Demonstrated

This project covers several important concepts:

* Data collection and loading
* Exploratory Data Analysis
* Missing-value handling
* Categorical feature encoding
* Feature-target separation
* Stratified train-test splitting
* Support Vector Machines
* Linear classification
* Model training
* Prediction
* Accuracy-based evaluation

---

## 📁 Project Structure

```text
Loan-Approval-Prediction/
│
├── Loan_approval_code.ipynb
├── dataset.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

Install the required dependencies:

```bash
pip install numpy pandas seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Loan_approval_code.ipynb
```

---

## 🚀 Future Improvements

The current implementation provides a basic SVM-based classification pipeline. Possible improvements include:

* Compare SVM with Logistic Regression, Random Forest, and XGBoost
* Use cross-validation for more reliable evaluation
* Add precision, recall, F1-score, and ROC-AUC
* Generate a confusion matrix
* Perform feature scaling before SVM training
* Perform hyperparameter tuning
* Build an interactive Streamlit prediction interface
* Add a dedicated prediction function for new applicants
* Improve handling of missing values instead of simply dropping rows

---

## 📚 What I Learned

Through this project, I practiced building an end-to-end supervised machine-learning pipeline, from **raw data preprocessing and visualization to model training and evaluation**.

The project also helped reinforce practical concepts such as categorical encoding, stratified data splitting, classification, and model performance evaluation using Scikit-learn.

---

## 👨‍💻 Author

**Olivia Gijo**

Master's Student in Information & Communication Technology
FAU Erlangen-Nürnberg, Germany

### Areas of Interest

* 🤖 Machine Learning
* 🧠 Deep Learning
* 📊 Data Science
* 🔊 Signal Processing
* 💬 NLP

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub!
