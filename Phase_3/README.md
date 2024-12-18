# Towards Personalized Perioperative Care: Predicting Duration of Hospitalization and Postoperative Complications

## **Group Members**
- **M V N S H Praneeth** - Person Number: 50592326  
- **Akhil V S S G** - Person Number: 50606819  
- **Abdul Wasi Lone** - Person Number: 50609995  

---

## **Datasets for Akhil's Models**
For the notebooks:  
- **Model_Training_akhil_Q_1_XGBoost**  
- **Model_Training_akhil_Q_2_CatBoost**

Please use the datasets provided in this link:  
**Datasets for Akhil's Models**:  
`Phase_3\app\Datasets\Processed_patient_post_op_complications_Final.csv`

---

## **Datasets for Praneeth's Models**
For the notebooks:  
- **M V N S H Praneeth_Phase 2 ANN**  
- **M V N S H Praneeth_Phase 2 Logistic Regression**

Please use the datasets provided in the same link:  
**Datasets for Praneeth's Models**:  
`Phase_3\app\Datasets\Patient_information_Final.csv`

---

## **Datasets for Wasi's Models**
For the notebook:  
- **Wasi_50609995_Models**

Please use the datasets provided in this link:  
**Datasets for Wasi's Models**:  
`Phase_3\app\Datasets\Patient_information_Final.csv`

---

## **Datasets**
We utilized the following datasets for our analysis and model training:

1. **Patient_Information.csv** - Contains demographic, procedural, and clinical details about each patient.  
2. **Post_op_complications.csv** - Lists any postoperative complications recorded for patients.  

---

## **Folder Structure Information**
- **`app`**: Contains folders named Datasets, app_code and models. app_code folder consists of app.py file and requirements.txt file.  
- **`exp`**: Contains the folders named Phase_1 and Phase_2 consisting of Python codes, the questions used by team members, and the analysis of the questions.  
- **Root Folder**: Consists of the final report and the description video.  

---

## **Models Used, Their Description, and Location**

### **1. XGBoost (Extreme Gradient Boosting)**  
Located in:  
`50592326_50609995_50606819_phase_3\exp\Model_Training_akhil_Q_1_XGBoost.ipynb`

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
Located in:  
`50592326_50609995_50606819_phase_3\exp\Model_Training_akhil_Q_2_CatBoost.ipynb`

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

### **3. Decision Tree Classifier**  
Located in:  
`50592326_50609995_50606819_phase_3\exp\Wasi_50609995_Models.ipynb`

#### **Reasons for Selection:**
- **Interpretability**: Aligns well with medical decision-making processes.  
- **Versatility**: Handles both numerical and categorical data effectively.  
- **Non-Linear Relationships**: Captures complex relationships between features.  

---

### **4. Random Forest Classifier**  
Located in:  
`50592326_50609995_50606819_phase_3\exp\Wasi_50609995_Models.ipynb`

#### **Reasons for Selection:**
- **Reduced Overfitting**: Ensemble learning improves generalization.  
- **Feature Importance**: Provides robust rankings for feature significance.  
- **High-Dimensional Data**: Handles datasets with high feature complexity effectively.  

---

### **5. Anesthesia Type Prediction Model**

#### **Model Architecture:**
- Implemented a deep neural network using Keras.  
- Used multiple dense layers with SELU and ReLU activations.  
- Applied dropout for regularization to prevent overfitting.  

#### **Training:**
- Utilized RMSprop optimizer with a learning rate of 0.01.  
- Implemented early stopping to prevent overfitting.  
- Used categorical crossentropy as the loss function.  

#### **Evaluation:**
- Split data into training (80%) and testing (20%) sets.  
- Achieved 86% accuracy on the test set.  

---

## **Instructions to Build the App from Source Code**

### **Step 1: Setting Up the Environment**
1. **Create a Virtual Environment**  
   Open a terminal and navigate to the project directory. Then, create a virtual environment by running the following command:  
   ```bash
   python3 -m venv env
   ```

2. **Activate the Virtual Environment**  
   Once the virtual environment is created, activate it:  
   - For macOS/Linux:  
     ```bash
     source env/bin/activate
     ```  
   - For Windows:  
     ```bash
     .\env\Scripts\activate
     ```

3. **Install the Required Dependencies**  
   After activating the virtual environment, install all necessary dependencies by running:  
   ```bash
   pip install -r requirements.txt
   ```  
   This will install all the packages listed in the `requirements.txt` file.  

---

### **Step 2: Organizing the Project**
- **App Code and Models**:  
  Ensure that the application code and the pre-trained models are kept in their respective folders. The models are usually located in the `models` folder.  

- **Datasets**:  
  The datasets required for the application are in the `app` folder. You need to either:  
  - **Download the datasets** from the `datasets` folder and upload them into MongoDB Compass manually.  
  - Or, **upload them directly** to MongoDB using MongoDB Compass with the correct collection names, as specified in the app code.  

---

### **Step 3: Setting Up MongoDB**
1. **MongoDB Connection**  
   The app connects to MongoDB using the MongoClient. If you have MongoDB hosted on a local server or use MongoDB Atlas, ensure your connection string is correctly set in the code.  
   The connection string should look something like this:  
   ```bash
   mongodb+srv://<username>:<password>@cluster0.i20jf.mongodb.net/
   ```

2. **Uploading Datasets to MongoDB**  
   Upload the datasets into MongoDB Compass:  
   - Open **MongoDB Compass** and connect to your MongoDB instance.  
   - Upload the datasets into collections named as they are used in the code (e.g., `Patient_Information`, `Patient_Post_OP`).  

---

### **Step 4: Running the Application**
1. **Start the App**  
   With all dependencies installed and the virtual environment activated, you can run the app by using the following command:  
   ```bash
   streamlit run patient-management-system.py
   ```

2. **Operating the Functionalities**  
   The application includes several functionalities:  
   - **View/Edit Patient Data**: Search for a patient's data and make updates.  
   - **Add New Patient**: Add new patient records.  
   - **Delete Patient**: Delete patient records.  
   - **Predict Outcomes**: Predict outcomes like post-operative complications, discharge disposition, anesthesia type, and length of stay based on patient data.  
