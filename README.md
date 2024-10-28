# Ensembled Prediction of Patient's Length of Stay, Medications, and Postoperative Complications

## Group Members
- **M V N S H Praneeth** - Person Number: 50592326
- **Akhil V S S G** - Person Number: 50606819
- **Abdul Wasi Lone** - Person Number: 50609995

---

## Project Overview
This project uses an ensemble approach to predict a patient's length of stay, required medications, and potential postoperative complications. Our goal is to support healthcare providers in improving resource allocation and enhancing patient care by analyzing and predicting these critical outcomes.

## Datasets
We utilized the following datasets for our analysis and model training:

1. **Patient_Information.csv** - Contains demographic, procedural, and clinical details about each patient.
2. **Patient_Coding.csv** - Provides diagnostic and procedural codes for patient encounters.
3. **Post_op_complications.csv** - Lists any postoperative complications recorded for patients.

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

---


