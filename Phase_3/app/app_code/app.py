import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import pickle
import os
from pathlib import Path
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from joblib import load


PRIMARY_ANES_TYPE_NM = {
    "General": 2,
    "Monitored Anesthesia Care (MAC)": 6,
    "Regional": 7,
    "Moderate Sedation - by non-anesthesia staff only": 4,
    "Local": 3,
    "Choice Per Patient on Day of Surgery": 0,
    "Epidural": 1,
    "Spinal": 8,
    "Topical": 9,
    "Monitored Anesthesia Care (MAC)": 5
}


COLUMN_PAIRS = {
    'ASA_RATING_C': 'ASA_RATING',
    'PRIMARY_ANES_TYPE_NM_ENCODED': 'PRIMARY_ANES_TYPE_NM',
    'DISCH_DISP_C': 'DISCH_DISP',
    'PATIENT_CLASS_NM_ENCODED': 'PATIENT_CLASS_NM',
    'PATIENT_CLASS_GROUP_ENCODED': 'PATIENT_CLASS_GROUP'
}


DISCH_DISP = {
    "Home Routine": 15,
    "Hospice Facility": 16,
    "Skilled Nursing Facility": 6,
    "Home Healthcare IP Admit Related": 20,
    "Rehab Facility (this hospital)": 100,
    "Expired": 3,
    "Acute Care Facility (not this hospital)": 26,
    "Jail/Prison": 10,
    "Long Term Care Facility": 30,
    "Rehab Facility (not this hospital)": 4,
    "Psychiatric Facility (this hospital)": 19,
    "Cancer Ctr/Children's Hospital": 18,
    "Acute Care Facility (this hospital)": 8,
    "Against Medical Advice": 13,
    "Hospice Home": 22,
    "Federal Hospital": 11,
    "Coroner": 23,
    "Other Healthcare Not Defined in this List": 70,
    "Sub-Acute Care Facility": 107,
    "Recuperative Care": 105,
    "Board and Care": 103,
    "Home Healthcare Outside 3 Days": 21,
    "Intermediate/Residential Care w Planned Readmit": 84,
    "Psychiatric Facility (not this hospital)": 9,
    "Shelter": 102,
    "Critical Access Hospital": 66,
    "Home Health w Planned Readmit": 86,
    "Designated Disaster Alternative Care Site": 69,
    "Intermediate/Residential Care Facility": 5,
    "Home Healthcare Outpatient Related": 109,
    "Federal Hospital w Planned Readmit": 88,
    "Designated Disaster Alternate Care Site": 108,
    "Temporary Living": 106,
    "Independent Living": 104,
    "Room and Board": 104
}

ASA_RATING = {
    "Severe Systemic Disease": 3,
    "Mild Systemic Disease": 2,
    "Healthy": 1,
    "Incapacitating Disease": 4,
    "Moribund": 5,
    "Brain Dead": 6
}

PATIENT_CLASS_GROUP = {
    "Outpatient": 1,
    "Inpatient": 0
}

PATIENT_CLASS_NM = {
    "Hospital Outpatient Surgery": 1,
    "Hospital Inpatient Surgery": 0,
    "Inpatient Admission": 2
}

ICU_ADMIN_FLAG = {
    "No": 0,
    "Yes": 1
}

SEX = {
    "Female": 0,
    "Male": 1
}

Post_Operative = {7:'Post_OP_type_AN  ADMINISTRATIVE',
 2:'Post_OP_type_AN  AIRWAY',
 10:'Post_OP_type_AN  CARDIOVASCULAR',
 8 : 'Post_OP_type_AN  INJURY/INFECTION',
 9:'Post_OP_type_AN  MEDICATION',
 6 :'Post_OP_type_AN  METABOLIC',
 3:'Post_OP_type_AN  NEUROLOGICAL',
 0:'Post_OP_type_AN AQI',
 1:'Post_OP_type_ANE  OTHER',
 4:'Post_OP_type_ANE  REGIONAL',
 5:'Post_OP_type_ANE  RESPIRATORY'}


NUMERIC_FIELDS = {'LOS', 'WEIGHT', 'HEIGHT_METRES', 'OR_LOS_HOURS'}

FIELD_ENCODINGS = {
    'PRIMARY_ANES_TYPE_NM_ENCODED': PRIMARY_ANES_TYPE_NM,
    'DISCH_DISP_C': DISCH_DISP,
    'ASA_RATING_C': ASA_RATING,
    'PATIENT_CLASS_GROUP_ENCODED': PATIENT_CLASS_GROUP,
    'PATIENT_CLASS_NM_ENCODED': PATIENT_CLASS_NM,
    'ICU_ADMIN_FLAG': ICU_ADMIN_FLAG,
    'SEX': SEX
}


REVERSE_COLUMN_PAIRS = {v: k for k, v in COLUMN_PAIRS.items()}

def get_dropdown_options(field):
    """Get dropdown options for a field, whether it's encoded or original"""
    if field in FIELD_ENCODINGS:
        return list(FIELD_ENCODINGS[field].keys())
    elif field in REVERSE_COLUMN_PAIRS:
        encoded_field = REVERSE_COLUMN_PAIRS[field]
        return list(FIELD_ENCODINGS[encoded_field].keys())
    return None

def get_key_from_value(dictionary, value):
    """Get the key (display value) from the encoded value"""
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def get_encoded_value(dictionary, key):
    """Get the encoded value from the display value"""
    return dictionary.get(key)


@st.cache_resource
def init_connection():
    try:
        client = MongoClient("mongodb+srv://proneeth4:Radhika%401969@cluster0.i20jf.mongodb.net/")
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

def get_database():
    try:
        client = init_connection()  
        if client is not None:
            return client['DIC_PROJECT']
        return None
    except Exception as e:
        st.error(f"Failed to get database: {str(e)}")
        return None

def get_collection(collection_type):
    try:
        db = get_database()
        if db is not None:
            if collection_type == "Patient Information":
                return db['Patient_Information']
            else:
                return db['Patient_Post_OP']
        return None
    except Exception as e:
        st.error(f"Failed to get collection: {str(e)}")
        return None
    

def load_model(model_name):
    """Load the ML model from the models directory"""
    try:
        current_dir = Path(__file__).parent
        model_path = current_dir / 'models' / model_name
        
        if not model_path.exists():
            st.error(f"Model file {model_name} not found in models folder")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def validate_numeric_input(value, field_name):
    if not value:
        return True
    try:
        float_value = float(value)
        return True
    except ValueError:
        st.error(f"{field_name} must be a numeric value")
        return False

def process_form_data(form_data, collection_type):
    """Process form data before saving to database"""
    processed_data = {}
    for field, value in form_data.items():
        if field in FIELD_ENCODINGS:
            # Handle encoded fields
            encoded_value = get_encoded_value(FIELD_ENCODINGS[field], value)
            processed_data[field] = encoded_value
        elif field in REVERSE_COLUMN_PAIRS:
            # Handle original fields - store display value
            processed_data[field] = value
        else:
            processed_data[field] = value
    return processed_data

def decode_db_data(db_data):
    """Convert encoded values to display values for showing to user"""
    decoded_data = {}
    for field, value in db_data.items():
        if field in FIELD_ENCODINGS:
            # Handle encoded fields
            display_value = get_key_from_value(FIELD_ENCODINGS[field], value)
            decoded_data[field] = display_value if display_value is not None else value
        elif field in REVERSE_COLUMN_PAIRS:
            # Handle original fields
            decoded_data[field] = value
        else:
            decoded_data[field] = value
    return decoded_data
def prepare_prediction_data(input_data):
    encoded_data = {}
    for field, value in input_data.items():
        if field in FIELD_ENCODINGS:
            encoded_data[field] = FIELD_ENCODINGS[field][value]
        else:
            encoded_data[field] = value
    return encoded_data

def view_edit_patient():
    st.header("View/Edit Patient Data")
    
    collection_type = st.selectbox(
        "Select Information Type",
        ["Patient Information", "Post-operative Information"]
    )
    
    collection = get_collection(collection_type)
    if collection is not None:
        st.subheader("Search Patient")
        mrn = st.text_input("Enter MRN to search")
        if mrn:
            patient_data = collection.find_one({"MRN": mrn})
            if patient_data:
                st.success("Patient found!")
                edit_patient_form(collection, patient_data)
            else:
                st.error("Patient not found!")

def edit_patient_form(collection, patient_data):
    with st.form("edit_form"):
        edited_data = {}
        
        patient_data.pop('_id', None)
        decoded_data = decode_db_data(patient_data)
        
        for field, value in decoded_data.items():
            if field in FIELD_ENCODINGS or field in REVERSE_COLUMN_PAIRS:

                options = get_dropdown_options(field)
                if options:
                    try:
                        current_index = options.index(value) if value in options else 0
                    except (ValueError, TypeError):
                        current_index = 0
                        
                    edited_data[field] = st.selectbox(
                        f"{field}",
                        options=options,
                        index=current_index
                    )
            elif field in NUMERIC_FIELDS:
                edited_data[field] = st.number_input(
                    f"{field}", 
                    value=float(value) if value else 0.0,
                    step=0.1
                )
            else:
                edited_data[field] = st.text_input(f"{field}", value)

        submit = st.form_submit_button("Save Changes")
        
        if submit:
            valid_numerics = all(
                validate_numeric_input(edited_data[field], field)
                for field in edited_data.keys()
                if field in NUMERIC_FIELDS
            )
            
            if valid_numerics:
                processed_data = process_form_data(edited_data, collection.name)

                collection.update_one(
                    {"MRN": patient_data['MRN']},
                    {"$set": processed_data}
                )
                st.success("Patient data updated successfully!")


def get_post_op_type_columns(collection):
    """Get all columns starting with Post_OP_type"""
    sample_doc = collection.find_one()
    if sample_doc:
        return [col for col in sample_doc.keys() if col.startswith('Post_OP_type')]
    return []

def add_new_patient():
    st.header("Add New Patient")
    
    collection_type = st.selectbox(
        "Select Information Type",
        ["Patient Information", "Post-operative Information"]
    )
    
    collection = get_collection(collection_type)
    if collection is not None:

        sample_doc = collection.find_one()
        if not sample_doc:
            st.warning(f"No existing documents found in {collection_type} collection. Creating form with basic fields.")
            sample_doc = {"MRN": ""}
        
        with st.form("add_form"):
            new_data = {}
            
            st.subheader(f"Enter {collection_type} Details")

            new_data['MRN'] = st.text_input("MRN (Required)")
            

            processed_fields = {'MRN', '_id'}

            for field in FIELD_ENCODINGS.keys():
                if field in sample_doc:
                    encoding_dict = FIELD_ENCODINGS[field]
                    new_data[field] = st.selectbox(
                        f"{field}",
                        options=list(encoding_dict.keys())
                    )
                    processed_fields.add(field)
                    

                    if field in COLUMN_PAIRS:
                        original_field = COLUMN_PAIRS[field]
                        new_data[original_field] = new_data[field]
                        processed_fields.add(original_field)

            for field in NUMERIC_FIELDS:
                if field in sample_doc and field not in processed_fields:
                    new_data[field] = st.number_input(f"{field}", min_value=0.0, step=0.1)
                    processed_fields.add(field)

            if collection_type == "Post-operative Information":
                post_op_types = get_post_op_type_columns(collection)
                if post_op_types:
                    st.subheader("Post-Operative Type")
                    selected_type = st.selectbox(
                        "Select Post-Operative Type",
                        options=post_op_types
                    )
                    
                    for field in post_op_types:
                        new_data[field] = 0
                    

                    new_data[selected_type] = 1
                    processed_fields.update(post_op_types)

            for field in sample_doc.keys():
                if field not in processed_fields and not field.startswith('Post_OP_type'):
                    new_data[field] = st.text_input(f"{field}")
            
            submit = st.form_submit_button("Add Patient")
            
            if submit:
                if not new_data['MRN']:
                    st.error("MRN is required!")
                elif collection.find_one({"MRN": new_data['MRN']}):
                    st.error("MRN already exists!")
                else:
                    valid_numerics = all(
                        validate_numeric_input(new_data[field], field)
                        for field in new_data.keys()
                        if field in NUMERIC_FIELDS
                    )
                    
                    if valid_numerics:
                        processed_data = process_form_data(new_data, collection_type)
                        collection.insert_one(processed_data)
                        st.success(f"New patient added successfully to {collection_type} collection!")

def delete_patient():
    st.header("Delete Patient")
    
    collection_type = st.selectbox(
        "Select Information Type",
        ["Patient Information", "Post-operative Information"]
    )
    
    collection = get_collection(collection_type)
    if collection is not None:
        st.subheader("Search Patient to Delete")
        mrn = st.text_input("Enter MRN to delete")
        if mrn:
            patient_data = collection.find_one({"MRN": mrn})
            if patient_data:
                st.write("Patient found in", collection_type)
                display_data = decode_db_data(patient_data.copy())
                del display_data['_id']
                st.write(display_data)
                
                with st.form("delete_form"):
                    submit = st.form_submit_button("Delete Patient")
                    if submit:
                        collection.delete_one({"MRN": mrn})
                        st.success(f"Patient deleted successfully from {collection_type} collection!")
            else:
                st.error(f"Patient not found in {collection_type} collection!")


def get_los_range(prediction):
    """Convert numeric prediction to LOS range string"""
    ranges = {
        1: "0-5 days",
        2: "5-10 days",
        3: "10-15 days",
        4: "15-20 days",
        5: "20-25 days",
        6: "25-30 days"
    }
    return ranges.get(prediction, "Unknown range")

def get_binned_value(value):
    """Convert numeric value to binned category"""
    bin_edges = [0, 5, 10, 15, 20, 25, 30]
    bin_labels = [1, 2, 3, 4, 5, 6]
    
    for i in range(len(bin_edges)-1):
        if bin_edges[i] <= value < bin_edges[i+1]:
            return bin_labels[i]
    return bin_labels[-1] if value >= bin_edges[-1] else bin_labels[0]

def predict_length_of_stay():
    st.subheader("Predict Length of Stay")
    
    with st.form("los_form"):

        age = st.number_input('Age', min_value=0, max_value=120, step=1)
        an_los_hours = st.number_input('Anesthesia Length of Stay (hours)', min_value=0.0, step=0.1)
        
        icu_admin = st.selectbox('ICU Admin Flag', options=list(ICU_ADMIN_FLAG.keys()))
        
        predict_button = st.form_submit_button("Predict Length of Stay")
        
        if predict_button:
            if age <= 0 or an_los_hours < 0:
                st.error("Please enter valid values for all numeric fields.")
                return
            

            binned_los = get_binned_value(an_los_hours)
            
            input_data = {
                'BIRTH_DATE': age,
                'AN_LOS_HOURS_BINNED': binned_los,
                'ICU_ADMIN_FLAG': icu_admin
            }
            
            encoded_data = prepare_prediction_data(input_data)
            input_df = pd.DataFrame([encoded_data])
            
            model = load('models/decision_tree.joblib')
            if model:
                try:
                    prediction = model.predict(input_df)
                    predicted_range = get_los_range(prediction[0])
                    st.success("Prediction Complete!")
                    st.write("### Predicted Length of Stay:")
                    st.write(predicted_range)
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

def predict_discharge():
    st.subheader("Predict Patient Discharge Disposition")
    
    with st.form("discharge_form"):
        # Dropdown inputs
        icu_admin = st.selectbox('ICU Admin Flag', options=list(ICU_ADMIN_FLAG.keys()))
        sex = st.selectbox('Sex', options=list(SEX.keys()))
        
        # Numeric inputs
        weight = st.number_input('Weight (Ounces)', min_value=0.0, step=0.1)
        height = st.number_input('Height (meters)', min_value=0.0, step=0.01)
        los = st.number_input('Length of Stay (days)', min_value=0.0, step=0.1)
        or_los_hours = st.number_input('OR Length of Stay (hours)', min_value=0.0, step=0.1)
        
        predict_button = st.form_submit_button("Predict Discharge Disposition")
        
        if predict_button:
            if weight <= 0 or height <= 0 or los <= 0 or or_los_hours <= 0:
                st.error("Please enter valid positive values for all numeric fields.")
                return
            
            input_data = {
                'LOS': los,
                'ICU_ADMIN_FLAG': icu_admin,
                'WEIGHT': weight,
                'SEX': sex,
                'OR_LOS_HOURS': or_los_hours,
                'HEIGHT_METRES': height
            }
            
            encoded_data = prepare_prediction_data(input_data)
            input_df = pd.DataFrame([encoded_data])
            model = CatBoostClassifier()
            model.load_model('models/catboost_model.cbm')
            if model:
                try:
                    prediction = model.predict(input_df)
                    st.success("Prediction Complete!")
                    st.write("### Predicted Discharge Disposition:")
                    disposition = get_key_from_value(DISCH_DISP, prediction[0])
                    st.write(disposition if disposition else "Unknown Disposition")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

def predict_anesthesia():
    st.subheader("Predict Anesthesia Type")
    
    with st.form("anesthesia_form"):

        disch_disp = st.selectbox('Discharge Disposition', options=list(DISCH_DISP.keys()))
        icu_admin = st.selectbox('ICU Admin Flag', options=list(ICU_ADMIN_FLAG.keys()))
        sex = st.selectbox('Sex', options=list(SEX.keys()))
        patient_class = st.selectbox('Patient Class', options=list(PATIENT_CLASS_NM.keys()))
        patient_group = st.selectbox('Patient Group', options=list(PATIENT_CLASS_GROUP.keys()))
        

        weight = st.number_input('Weight (kg)', min_value=0.0, step=0.1)
        height = st.number_input('Height (meters)', min_value=0.0, step=0.01)
        los = st.number_input('Length of Stay (days)', min_value=0.0, step=0.1)
        
        predict_button = st.form_submit_button("Predict Anesthesia Type")
        
        if predict_button:
            if weight <= 0 or height <= 0 or los <= 0:
                st.error("Please enter valid positive values for all numeric fields.")
                return
            
            input_data = {
                'DISCH_DISP_C': disch_disp,
                'LOS': los,
                'ICU_ADMIN_FLAG': icu_admin,
                'WEIGHT': weight,
                'SEX': sex,
                'PATIENT_CLASS_NM_ENCODED': patient_class,
                'PATIENT_CLASS_GROUP_ENCODED': patient_group,
                'HEIGHT_METRES': height
            }
            
            encoded_data = prepare_prediction_data(input_data)
            input_df = pd.DataFrame([encoded_data])
            
            model = load_model('trained_model_ANN.pkl')
            if model:
                try:
                    prediction = model.predict(input_df)
                    st.success("Prediction Complete!")
                    st.write("### Predicted Anesthesia Type:")
                    anesthesia_type = get_key_from_value(PRIMARY_ANES_TYPE_NM, prediction.argmax(axis=1))
                    st.write(anesthesia_type if anesthesia_type else "Unknown Anesthesia Type")
                    # st.write()
                    # print(prediction)
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

def predict_complications():
    st.subheader("Predict Post Operative Complications")
    
    with st.form("complications_form"):
        icu_admin = st.selectbox('ICU Admin Flag', options=list(ICU_ADMIN_FLAG.keys()))
        sex = st.selectbox('Sex', options=list(SEX.keys()))
        asa_rating = st.selectbox('ASA Rating', options=list(ASA_RATING.keys()))
        anes_type = st.selectbox('Primary Anesthesia Type', options=list(PRIMARY_ANES_TYPE_NM.keys()))
        
        weight = st.number_input('Weight (kg)', min_value=0.0, step=0.1)
        height = st.number_input('Height (meters)', min_value=0.0, step=0.01)
        los = st.number_input('Length of Stay (days)', min_value=0.0, step=0.1)
        or_los_hours = st.number_input('OR Length of Stay (hours)', min_value=0.0, step=0.1)
        
        predict_button = st.form_submit_button("Predict Complications")
        
        if predict_button:
            if weight <= 0 or height <= 0 or los <= 0 or or_los_hours <= 0:
                st.error("Please enter valid positive values for all numeric fields.")
                return
            
            input_data = {
                'ICU_ADMIN_FLAG': icu_admin,
                'WEIGHT': weight,
                'SEX': sex,
                'ASA_RATING_C': asa_rating,
                'OR_LOS_HOURS': or_los_hours,
                'HEIGHT_METRES': height,
                'LOS': los,
                'PRIMARY_ANES_TYPE_NM_ENCODED': anes_type
            }
            
            encoded_data = prepare_prediction_data(input_data)
            input_df = pd.DataFrame([encoded_data])
            model = XGBClassifier()
            model.load_model('models/xgb_model.json')
            if model:
                try:
                    prediction = model.predict(input_df)
                    st.success("Prediction Complete!")
                    st.write("### Predicted Post-operative Complication:")
                    complication_type = Post_Operative.get(prediction[0], "Unknown Complication")
                    st.write(complication_type)
                    # st.write(prediction[0])
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")



def main():
    st.title("Patient Information Management System")
    
    operation = st.sidebar.radio(
        "Select Operation",
        ["View/Edit Patient Data", "Add New Patient", "Delete Patient", "Predict Outcomes"]
    )
    
    if operation == "View/Edit Patient Data":
        view_edit_patient()
    elif operation == "Add New Patient":
        add_new_patient()
    elif operation == "Delete Patient":
        delete_patient()
    elif operation == "Predict Outcomes":
        st.header("Outcome Prediction")
        prediction_type = st.selectbox(
            "Select Prediction Type",
            ["Post Operative Complications", "Patient Discharge Disposition", 
             "Anesthesia Type", "Length of Stay"]
        )
        
        if prediction_type == "Post Operative Complications":
            predict_complications()
        elif prediction_type == "Patient Discharge Disposition":
            predict_discharge()
        elif prediction_type == "Anesthesia Type":
            predict_anesthesia()
        elif prediction_type == "Length of Stay":
            predict_length_of_stay()

            
if __name__ == "__main__":
    main()