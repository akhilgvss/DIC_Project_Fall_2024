# Ensembled Prediction of Patient's Length of Stay, Medications, and Postoperative Complications

## Group Members
- **M V N S H Praneeth** - Person Number: 50592326
- **Akhil V S S G** - Person Number: 50606819
- **Abdul Wasi Lone** - Person Number: 50609995

---
## Dataset Overview
1. We aim to address how our method can be useful in cutting costs for the patients and increasing profit for the hospitals due to the optimal allocation of resources.

2. Proper anticipation of the duration of hospitalization, type of medication, and the anesthesia type is of utmost relevance to patients, hospitals, and the insurance companies, and is therefore relevant (popular).

3. The problem has a rich background since the dataset we took was from an exhaustive medical dataset (MOVER: Medical Informatics Operating Room Vitals and Events Repository) from UC Irvine having the hospitalization data of 58,799 patients and 83,468 surgeries (Aprrox. 500 GB). Link to paper: [Mover Dataset](https://www.medrxiv.org/content/10.1101/2023.03.03.23286777v2.full.pdf)
Login Credentals for the dataset : [Dataset site](https://mover-download.ics.uci.edu/) Username : akhilven , Password : fJqZSWLB02VQ3IAqF 

5. The integration of AI into these healthcare settings which ultimately leads to improvement of patient outcomes (like the length of stay) and Data driven personalized healthcare makes the work viable. Moreover the Optimized resource management as discussed in [Project Document](https://docs.google.com/document/d/17oPCBdx473Aap0shQprmMPKT3NHszan7x9u_0UJei6k/edit?usp=sharing). holds equal importance towards this direction.

---
# Dataset Usage Instructions

To ensure that the provided Jupyter Notebooks execute correctly, please use the corresponding datasets for each notebook as detailed below.

---

## **Datasets for Akhil's Models**
For the notebooks:
- **Model_Training_akhil_Q_1_XGBoost**
- **Model_Training_akhil_Q_2_CatBoost**

Please use the datasets provided in this Google Drive link:  
[Datasets for Akhil's Models](https://drive.google.com/drive/folders/1aVNenfwCIbxGtlvMMLr59axqp3AmYKy9?usp=sharing)

---

## **Datasets for Praneeth's Models**
For the notebooks:
- **M V N S H Praneeth_Phase 2 ANN**
- **M V N S H Praneeth_Phase 2 Logistic Regression**

Please use the datasets provided in the same Google Drive link:  
[Datasets for Praneeth's Models](https://drive.google.com/drive/folders/1aVNenfwCIbxGtlvMMLr59axqp3AmYKy9?usp=sharing)

---

## **Datasets for Wasi's Models**
For the notebook:
- **Wasi_50609995_Models**

Please use the datasets provided in this Google Drive link:  
[Datasets for Wasi's Models](https://drive.google.com/drive/folders/1_6aV6_Ji7RgQ7TQFVpHTIk4kFqd3Bi-W?usp=sharing)

---

## **Important Notes**
1. **File Path Configuration**: Ensure that you configure the file paths appropriately in the code for the datasets to load properly. Mismatched paths can lead to runtime errors.
2. **Directory Structure**: Organize your datasets in directories according to the provided paths or adjust the paths in the code as needed.
3. **Dataset Compatibility**: Verify that the dataset schema matches the requirements of the specific notebook you are executing.

By following these instructions, you will ensure that the code runs efficiently and effectively without any issues.

---
## Project Overview
This project uses an ensemble approach to predict a patient's length of stay, required medications, and potential postoperative complications. Our goal is to support healthcare providers in improving resource allocation and enhancing patient care by analyzing and predicting these critical outcomes.

## Datasets
We utilized the following datasets for our analysis and model training:

1. **Patient_Information.csv** - Contains demographic, procedural, and clinical details about each patient.
2. **Patient_Coding.csv** - Provides diagnostic and procedural codes for patient encounters.
3. **Post_op_complications.csv** - Lists any postoperative complications recorded for patients.
# Phase 1
## Data Cleaning and Preprocessing Steps for Patient_Information.csv

The data cleaning steps applied to the `Patient_Information.csv` file include:

1. **Length of Stay (LOS) Calculation**
   - Calculated the length of stay using `Admission time` and `Discharge time`.
   - Cross-checked the calculated values with the existing `LOS` column.
   - Dropped `Admission time` and `Discharge time` columns after validation.

2. **DISCH_DISP**  
   - Removed null values.
   - Encoded this column using label encoding to convert categories into numeric format.

3. **ICU ADMIN FLAG**  
   - Applied label encoding for categorical transformation.

4. **Surgery Date**  
   - Encoded the date and time fields to ensure compatibility with modeling processes.

5. **SEX**  
   - Used label encoding to convert gender data into numeric form.

6. **PRIMARY_ANES_TYPE_NM**  
   - Dropped rows with missing values.
   - Encoded values using label encoding.

7. **ASA_RATING**  
   - Removed rows with missing values and applied label encoding.

8. **PATIENT_CLASS_GRP and PATIENT_CLASS_NM**  
   - Performed label encoding on both columns.

9. **PRIMARY_PROCEDURE_NM**  
   - Tokenized values into tensors using BERT for deep learning models.

10. **OR Duration**  
    - Calculated as the time difference between `OR IN TIME` and `OR OUT TIME`.

11. **Anesthesia Duration**  
    - Computed using the time difference between `ANESTHESIA IN TIME` and `ANESTHESIA OUT TIME`.

12. **HEIGHT and WEIGHT**  
    - Plotted box plots to detect and remove outliers.
    - Converted height from feet/inches to meters.
    - Plotted density plots for height and filled missing values with the mean or median.

---

## Getting Started
To run this project, clone the repository, ensure all dependencies are installed, and follow the preprocessing steps outlined above.

## Requirements
- Python libraries: Pandas, NumPy, Scikit-Learn, TensorFlow/PyTorch (for BERT tokenization), Matplotlib/Seaborn (for plotting).


# Phase 2
# M V N S H Praneeth
# ICU Admission Prediction Model

## Features

- Utilizes key patient data: discharge disposition, length of stay, ASA rating, patient class group, and sex
- Implements feature scaling and polynomial feature generation
- Employs grid search for hyperparameter tuning
- Achieves 82% accuracy in predicting ICU admissions

## Implementation Details

1. **Data Preprocessing**: 
   - Selected relevant features
   - Applied standard scaling to normalize the data
   - Generated polynomial features to capture non-linear relationships

2. **Model Selection**: 
   - Chose Logistic Regression for its interpretability and efficiency
   - Used GridSearchCV to optimize hyperparameters

3. **Evaluation**: 
   - Split data into training (80%) and testing (20%) sets
   - Evaluated model using accuracy score, classification report, and confusion matrix

## Results

The optimized logistic regression model achieved an accuracy of 82%, demonstrating strong predictive power for ICU admissions based on the given features.

# Anesthesia Type Prediction Model

## Description

This project implements a deep learning model to predict the type of anesthesia used for patients based on various clinical factors, including discharge disposition, length of stay, ICU admission, and patient characteristics. The model uses a neural network architecture to capture complex relationships between the input variables and anesthesia type.

## Features

- Utilizes key patient data: discharge disposition, length of stay, ICU admission flag, weight, sex, patient class, height, and more
- Implements feature scaling for data normalization
- Employs a deep neural network with dropout layers for regularization
- Achieves 86% accuracy in predicting anesthesia type

## Implementation Details

1. **Data Preprocessing**: 
   - Selected relevant features
   - Applied standard scaling to normalize the data
   - Encoded the target variable (anesthesia type) using one-hot encoding

2. **Model Architecture**: 
   - Implemented a deep neural network using Keras
   - Used multiple dense layers with SELU and ReLU activations
   - Applied dropout for regularization to prevent overfitting

3. **Training**: 
   - Utilized RMSprop optimizer with a learning rate of 0.01
   - Implemented early stopping to prevent overfitting
   - Used categorical crossentropy as the loss function

4. **Evaluation**: 
   - Split data into training (80%) and testing (20%) sets
   - Achieved 86% accuracy on the test set

## **Akhil Venkata Shiva Sai**  
### **Person Number:** 50606819  

---

## Overview

This repository contains two machine learning models implemented to answer critical questions in healthcare, specifically focusing on post-operative trends and gender-based discharge and ICU admission analysis. The repository includes notebooks, graphs, and a detailed report documenting the methodologies, metrics, and results.

---

## Models Used

### **1. XGBoost (Extreme Gradient Boosting)**

#### **Data Preprocessing:**
- Merged datasets using unique keys created from `MRN_number` and `Log_ID`.
- Removed duplicate rows and handled imbalanced target classes using SMOTE.
- Selected critical features based on correlation and domain knowledge.

#### **Model Architecture:**
- XGBoost uses gradient boosting over decision trees, optimized for numerical and dense data.
- Hyperparameters tuned: `learning_rate`, `max_depth`, `n_estimators`.

#### **Training:**
- Trained on post-operative trends using features like `LOS`, `ICU_ADMIN_FLAG`, `WEIGHT`, and `ASA_RATING_C`.
- Used a 70-30 train-test split and 5-fold cross-validation for robustness.

#### **Evaluation:**
- Generated Precision-Recall and ROC-AUC curves to assess class separability.
- Used feature importance analysis to understand contributing factors.

#### **Metrics:**
- Macro and weighted averages of Precision, Recall, and F1-Score.
- AUC for multiclass classification demonstrates high performance.

---

### **2. CatBoost (Categorical Boosting)**

#### **Data Preprocessing:**
- Target column (`DISCH_DISP`) was label-encoded and balanced across classes.
- Key categorical features like `SEX` and `DISCH_DISP` were retained without extensive preprocessing, leveraging CatBoost's native categorical handling.

#### **Model Architecture:**
- CatBoost builds gradient-boosted trees optimized for categorical features.
- Hyperparameters tuned: `depth`, `iterations`, `learning_rate`.

#### **Training:**
- Trained on features like `SEX`, `LOS`, and `ICU_ADMIN_FLAG` to explore relationships between gender and ICU admissions.
- Handled imbalanced data using oversampling techniques.

#### **Evaluation:**
- Confusion matrix and feature importance graphs were generated for better interpretability.
- Training vs Validation Loss curve ensured consistent learning without overfitting.

#### **Metrics:**
- Macro averages of Precision, Recall, and F1-Score.
- Multiclass ROC-AUC demonstrates excellent performance across all categories.
---

## Relevance to Real-World Healthcare

1. **Cost Reduction for Patients:**  
   Predicting LOS and ICU admissions helps reduce patient expenses by optimizing resource allocation and reducing complications.

2. **Profitability for Hospitals:**  
   Efficient ICU and resource management lead to increased patient turnover and revenue generation.

3. **Insurance and Policy Implications:**  
   Predictions aid insurance companies in better risk assessment and policy-makers in healthcare funding.

4. **Integration of AI:**  
   - Personalized healthcare through predictive analytics.  
   - Optimized hospital operations using data-driven insights.

---

## Contents

- `Model_Training_akhil_Q_1_XGBoost.ipynb`: Notebook for XGBoost implementation.
- `Model_Training_akhil_Q_2_CatBoost.ipynb`: Notebook for CatBoost implementation.
- `DIC_Phase_2_report_akhil.pdf`: Comprehensive report detailing methodologies, metrics, and results.
- Supporting Graphs:
  - Training vs Validation Loss.
  - Feature Importance Graphs.
  - Precision-Recall and ROC-AUC Curves.
  - Confusion Matrices.

---
## Metrics

### **1. XGBoost**

| **Metric**      | **Value** |
|------------------|-----------|
| **Accuracy**     | 98%       |
| **Macro Precision** | 13%    |
| **Macro Recall**    | 14%    |
| **Macro F1-Score**  | 13%    |
| **Weighted Precision** | 98% |
| **Weighted Recall**    | 98% |
| **Weighted F1-Score**  | 98% |
| **AUC (Area Under Curve)** | Varies per class but exceeds 0.94 across most classes |

**Insights:**
- The **weighted metrics** are high, reflecting the model's strong performance for dominant classes.  
- **Macro metrics** are low due to class imbalance, highlighting the challenge of underrepresented classes.

---

### **2. CatBoost**

| **Metric**      | **Value** |
|------------------|-----------|
| **Accuracy**     | 96%       |
| **Macro Precision** | 88%    |
| **Macro Recall**    | 87%    |
| **Macro F1-Score**  | 87%    |
| **Weighted Precision** | 95% |
| **Weighted Recall**    | 96% |
| **Weighted F1-Score**  | 96% |
| **AUC (Area Under Curve)** | Varies per class, exceeding 0.92 for most classes |

**Insights:**
- The **macro metrics** demonstrate balanced performance across classes.  
- The ability to handle categorical features natively helped CatBoost perform well on **discharge disposition** analysis.

---

## Conclusion

The XGBoost and CatBoost models demonstrate the transformative potential of AI in healthcare, addressing critical patient outcomes and hospital resource challenges. By leveraging predictive analytics, this project aims to improve patient care and operational efficiency.

---
## Abdul Wasi Lone
### **Person Number:** 50609995

## **Introduction**
This study aims to identify factors that impact the **length of hospitalization for patients** and explore the **relationship between patient characteristics and ASA ratings**. We employed two machine learning algorithms, **Decision Tree Classifier** and **Random Forest Classifier**, to analyze a comprehensive medical dataset and derive actionable insights.

---

## **Dataset Overview**
The dataset contains various medical features, including:

- **BIRTH_DATE**: Patient age  
- **GENDER**: Gender of the patient  
- **ASA_RATING_C**: ASA physical status classification  
- **AN_TYPE**: Type of anesthesia  
- **ICU_ADMIN_FLAG**: ICU admission status  
- **AN_LOS_HOURS**: Length of stay in hours  

---

## **Methodology**

### **Data Preprocessing**
We implemented the following preprocessing steps:
1. **Categorical Variable Encoding**: Applied `OneHotEncoder` to encode categorical features.
2. **Numerical Feature Scaling**: Used `StandardScaler` for scaling numerical data.
3. **Data Splitting**: Split the dataset into **80% training** and **20% testing**.

### **Model Selection and Justification**

#### **1. Decision Tree Classifier**
We selected the Decision Tree Classifier due to:
- **Interpretability**: Aligns well with medical decision-making processes.  
- **Versatility**: Handles both numerical and categorical data effectively.  
- **Non-Linear Relationships**: Captures complex relationships between features.

#### **2. Random Forest Classifier**
We chose the Random Forest Classifier because:
- **Reduced Overfitting**: Ensemble learning improves generalization.  
- **Feature Importance**: Provides robust rankings for feature significance.  
- **High-Dimensional Data**: Handles datasets with high feature complexity effectively.

---

### **Model Tuning**

#### **Decision Tree Classifier**
We used **GridSearchCV** to tune the following parameters:
- `max_depth`: [3, 5, 7, 9]  
- `min_samples_split`: [2, 5, 10]  
- `min_samples_leaf`: [1, 2, 5, 10]  

**Best Parameters**:  
- `max_depth=5`, `min_samples_split=2`, `min_samples_leaf=5`

#### **Random Forest Classifier**
We used **GridSearchCV** to optimize these parameters:
- `n_estimators`: [100, 200]  
- `max_features`: ['auto', 'sqrt']  
- `max_depth`: [10, 20, 30, None]  

**Best Parameters**:  
- `n_estimators=200`, `max_features='sqrt'`, `max_depth=10`

---

## **Results and Analysis**

### **Decision Tree Classifier**
| **Metric**                        | **Value**                     |
|------------------------------------|-------------------------------|
| **AUC-ROC**                       | 0.9282                        |
| **Precision-Recall AUC (Macro)**  | 0.7190                        |
| **Accuracy**                      | 92.74%                        |

---

### **Random Forest Classifier**
| **Metric**                        | **Value**                     |
|------------------------------------|-------------------------------|
| **Best Hyperparameters**          | {'classifier_max_depth': 5, 'classifier_min_samples_split': 2, 'classifier_n_estimators': 100} |
| **Accuracy**                      | 92.74%                        |

