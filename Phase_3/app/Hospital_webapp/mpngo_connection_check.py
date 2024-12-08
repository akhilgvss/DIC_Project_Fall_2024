from pymongo import MongoClient
import pandas as pd

def test_mongodb_connection():
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb+srv://proneeth4:Radhika%401969@cluster0.i20jf.mongodb.net/")
        print("MongoDB Connection Successful!")
        
        # Access the database
        db = client['DIC_PROJECT']
        print("\nConnected to database:", db.name)
        
        # List all collections
        collections = db.list_collection_names()
        print("\nAvailable collections:", collections)
        
        # Test Patient_Information collection
        patient_info = db['Patient_Information']
        patient_count = patient_info.count_documents({})
        print(f"\nNumber of documents in Patient_Information: {patient_count}")
        
        # Print first document from Patient_Information
        print("\nSample document from Patient_Information:")
        first_doc = patient_info.find_one()
        if first_doc:
            print(first_doc)
        
        # Test Patient_Post_OP collection
        post_op = db['Patient_Post_OP']
        post_op_count = post_op.count_documents({})
        print(f"\nNumber of documents in Patient_Post_OP: {post_op_count}")
        
        # Print first document from Patient_Post_OP
        print("\nSample document from Patient_Post_OP:")
        first_post_op = post_op.find_one()
        if first_post_op:
            print(first_post_op)
        
        return True
        
    except Exception as e:
        print("\nError connecting to MongoDB:", str(e))
        return False

if __name__ == "__main__":
    print("Testing MongoDB Connection...")
    test_mongodb_connection()

# print('hi')