# 🎓 Student Academic Performance Prediction

A Machine Learning and Deep Learning-based application that predicts student academic performance using demographic, academic, social, and behavioural attributes.

## 📌 Project Overview

This project aims to analyse student-related factors and predict their final academic performance using classification techniques.

The system uses the **UCI Student Performance Dataset (Portuguese dataset)** and implements two classification models:

- Random Forest Classifier
- Deep Neural Network (DNN) using PyTorch

Students are classified into three performance categories: At Risk, Average, and High Performer.

An interactive web application built with Streamlit allows users to enter student details, explore dataset insights, evaluate model performance, and generate predictions using both models.

## 🎯 Objectives

- Analyse student academic performance and related attributes.
- Perform Exploratory Data Analysis (EDA) to identify patterns and relationships.
- Develop machine learning and deep learning classification models.
- Compare the performance of Random Forest and DNN.
- Build an interactive application for student performance prediction.

## 📊 Dataset

**Source:** UCI Machine Learning Repository

**Dataset:** Student Performance (Portuguese)

**Dataset ID:** 320

- Total student records: 649
- Original input features: 30
- Additional academic features: G1 and G2
- Target variable: G3 (Final Grade)

The final grade (G3) is converted into three classification categories:

| Performance Category | Grade Range |
|---|---|
| At Risk | 0–9 |
| Average | 10–14 |
| High Performer | 15–20 |

G1 and G2 are included as input features to help predict the final grade category.

## 🔍 Exploratory Data Analysis

EDA is performed to understand the dataset, feature distributions, and relationships between student attributes and final grades.

The analysis includes:

- Dataset information and descriptive statistics.
- Missing value analysis.
- Performance category distribution.
- Study time vs. average final grade.
- Previous failures vs. average final grade.
- Alcohol consumption vs. average final grade.
- Age vs. average final grade.
- Correlation analysis with the final grade.

Bar charts, line charts, and correlation tables are used to present the findings.

## ⚙️ Data Preprocessing

The following preprocessing steps are implemented:

1. **Target Creation:** Convert numerical final grades into three performance categories.
2. **Feature Selection:** Use demographic, social, behavioural, and academic features, including G1 and G2.
3. **Train-Test Split:** Split the dataset into 80% training and 20% testing data using stratified sampling.
4. **Categorical Encoding:** Apply OneHotEncoder to convert categorical features into numerical representations.
5. **Feature Scaling:** Apply StandardScaler to standardise numerical features.
6. **Feature Combination:** Combine encoded categorical and scaled numerical features into the final model input.

The encoder and scaler are fitted only on the training data and then applied to the testing data to prevent data leakage.

After preprocessing, the input feature dimension is 58.

## 🤖 Machine Learning Model – Random Forest

A Random Forest Classifier is implemented as the traditional machine learning model.

Configuration:
- Algorithm: Random Forest Classifier
- Number of estimators: 100
- Random state: 42

The model is trained using the preprocessed training data and evaluated on the testing data.

## 🧠 Deep Learning Model – DNN

A Deep Neural Network is developed using PyTorch for multi-class classification.

### Network Architecture

| Layer | Configuration |
|---|---|
| Input Layer | 58 features |
| Hidden Layer 1 | 64 neurons + ReLU |
| Hidden Layer 2 | 32 neurons + ReLU |
| Output Layer | 3 neurons |

### Training Configuration

- Framework: PyTorch
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 100
- Random Seed: 42

The network uses forward propagation to generate predictions and backpropagation to update its weights during training.

## 📈 Model Evaluation

Both models are evaluated using the following classification metrics:

- **Accuracy:** Measures the overall proportion of correct predictions.
- **Precision:** Measures how many predicted instances of a class are actually correct.
- **Recall:** Measures how many actual instances of a class are correctly identified.
- **F1-Score:** Provides a balance between precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions across the three categories.

Macro-averaged precision, recall, and F1-score are also used to compare performance across classes.

A comparison table is provided to evaluate Random Forest against the Deep Neural Network.

## 💻 Streamlit Web Application

The project includes an interactive dashboard with four main sections:

1. **Dashboard:** Displays dataset overview, total student records, and performance category distribution.
2. **Exploratory Data Analysis:** Presents statistical information, visualisations, and feature relationships.
3. **Model Evaluation:** Displays DNN classification metrics, confusion matrix, and Random Forest vs. DNN comparison.
4. **Student Prediction:** Allows users to enter student details and obtain predictions from both classification models.

The same trained preprocessing objects are used to transform user inputs before generating predictions.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| Scikit-learn | Data preprocessing, Random Forest, and evaluation |
| PyTorch | Deep Neural Network development and training |
| Streamlit | Interactive web application |
| SciPy | Combining sparse and dense feature matrices |
| UCI ML Repository | Dataset source |

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd Student-Performance-Prediction
```

### 3. Install dependencies

```bash
pip install torch streamlit ucimlrepo pandas scikit-learn scipy
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your default web browser.

**Note:** An internet connection is required to retrieve the dataset from the UCI Machine Learning Repository when running the application.

## 🔮 Future Improvements

- Improve model performance through hyperparameter tuning.
- Experiment with additional machine learning and deep learning architectures.
- Add more detailed prediction insights.
- Improve application deployment and scalability.
- Explore additional student-related features for better analysis.

## 👨‍💻 Author

**Aditya Kashyap**

Computer Science Engineering – Artificial Intelligence and Machine Learning

---

*This project demonstrates the application of Machine Learning and Deep Learning techniques to educational data analysis and academic performance prediction.*
