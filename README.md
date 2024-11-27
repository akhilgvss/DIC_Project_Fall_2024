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



