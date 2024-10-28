 Title - Ensembled Prediction of Patient's Length of Stay, Medications, and Postoperative Complications

Group Members:
M V N S H Praneeth - Person Number: 50592326
Akhil V S S G - Person Number: 50606819
Abdul Wasi Lone = Person Number: 50609995

Datasets:
Patient_Information.csv
Patient_Coding.csv
Post_op_complications.csv

Cleaning Steps Performed in Patient_Information.csv
Calculates Length of Stay and cross checked it with LOS column. Using Admission time and Discharge time.
Remove Admission time and Discharge time
DISCH_DISP - Label encoding and removed null values
ICU ADMIN FLAG - Label encoding
Surgery date - encode date and time
SEX column - Label encoding
PRIMARY_ANES_TYPE_NM - dropped nan, Label encoding
ASA_RATING - droped nan values and performed Label encoding
PATIENT_CLASS_GRP - Performed Label encoding
PATIENT_CLASS_NM - Performed Label encoding
PRIMARY_PROCEDURE_NM -  tokenize into tensors using BERT
OR IN TIME - OR - OUT TIME - Create OR Duration
ANASTHESIA IN TIME - ANESTHESIA OUT TIME - Create Anesthesia Duration
HEIGHT AND WEIGHT -  Plotted box plots for Both of the Columns and removed Outliers, Converted the Height column from feet/inches into metres. Plotting the density plot of Height column and filled null values with mean or median.
