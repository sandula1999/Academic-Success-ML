import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open('catboost_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_input(data):
    processed_data = np.array(data).reshape(1, -1) 
    return processed_data

def predict_student_status(input_data):
    processed_data = preprocess_input(input_data)
    
    if processed_data.shape[1] != 37:
        raise ValueError(f"Expected 37 features, but got {processed_data.shape[1]} features.")
    
    prediction = model.predict(processed_data)
    
    status_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    return status_mapping[int(prediction[0])]

st.title("🎓 Student Academic Success Predictor")

st.write(
    """
    Use this tool to predict a student's academic success based on a set of input features. You can manually enter the features or upload a CSV file.
    """
)

st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose Input Method", ("Manual Input", "Upload CSV"))

# --- Manual Input Section ---
if option == "Manual Input":
    st.header("🔧 Manually Input Student Features")

    # Collect ID from user (this will be included in the model prediction as the first feature)
    student_id = st.text_input("Student ID", placeholder="Enter student ID...")  # This will be included in the input to the model

    st.write("### Enter the following student details:")
    col1, col2, col3 = st.columns(3)

    with col1:
        marital_status = st.selectbox("Marital Status", options=[1, 2])  # Feature 0
        app_mode = st.number_input("Application Mode", min_value=1, max_value=50)  # Feature 1
        app_order = st.number_input("Application Order", min_value=1, max_value=5)  # Feature 2
        course = st.number_input("Course", min_value=9000, max_value=10000)  # Feature 3
        attendance = st.selectbox("Daytime/Evening Attendance", options=[1, 2])  # Feature 4
        prev_qualification = st.number_input("Previous Qualification", min_value=1, max_value=50)  # Feature 5

    with col2:
        prev_qualification_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, step=0.1)  # Feature 6
        nationality = st.number_input("Nationality", min_value=1, max_value=200)  # Feature 7
        mother_qualification = st.number_input("Mother's Qualification", min_value=1, max_value=50)  # Feature 8
        father_qualification = st.number_input("Father's Qualification", min_value=1, max_value=50)  # Feature 9
        mother_occupation = st.number_input("Mother's Occupation", min_value=1, max_value=50)  # Feature 10
        father_occupation = st.number_input("Father's Occupation", min_value=1, max_value=50)  # Feature 11

    with col3:
        admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, step=0.1)  # Feature 12
        displaced = st.selectbox("Displaced", options=[0, 1])  # Feature 13
        special_needs = st.selectbox("Educational Special Needs", options=[0, 1])  # Feature 14
        debtor = st.selectbox("Debtor", options=[0, 1])  # Feature 15
        tuition_fees = st.selectbox("Tuition Fees Up to Date", options=[0, 1])  # Feature 16
        gender = st.selectbox("Gender", options=[0, 1])  # Feature 17

    st.write("### Academic Information")
    col4, col5, col6 = st.columns(3)

    with col4:
        scholarship = st.selectbox("Scholarship Holder", options=[0, 1])  # Feature 18
        age = st.number_input("Age at Enrollment", min_value=17, max_value=60)  # Feature 19
        international = st.selectbox("International Student", options=[0, 1])  # Feature 20
        sem1_credited = st.number_input("Curricular Units 1st Sem Credited", min_value=0, max_value=10)  # Feature 21
        sem1_enrolled = st.number_input("Curricular Units 1st Sem Enrolled", min_value=0, max_value=10)  # Feature 22
        sem1_eval = st.number_input("Curricular Units 1st Sem Evaluations", min_value=0, max_value=10)  # Feature 23

    with col5:
        sem1_approved = st.number_input("Curricular Units 1st Sem Approved", min_value=0, max_value=10)  # Feature 24
        sem1_grade = st.number_input("Curricular Units 1st Sem Grade", min_value=0.0, max_value=20.0, step=0.1)  # Feature 25
        sem1_without_eval = st.number_input("Curricular Units 1st Sem Without Evaluations", min_value=0, max_value=10)  # Feature 26
        sem2_credited = st.number_input("Curricular Units 2nd Sem Credited", min_value=0, max_value=10)  # Feature 27
        sem2_enrolled = st.number_input("Curricular Units 2nd Sem Enrolled", min_value=0, max_value=10)  # Feature 28

    with col6:
        sem2_eval = st.number_input("Curricular Units 2nd Sem Evaluations", min_value=0, max_value=10)  # Feature 29
        sem2_approved = st.number_input("Curricular Units 2nd Sem Approved", min_value=0, max_value=10)  # Feature 30
        sem2_grade = st.number_input("Curricular Units 2nd Sem Grade", min_value=0.0, max_value=20.0, step=0.1)  # Feature 31
        sem2_without_eval = st.number_input("Curricular Units 2nd Sem Without Evaluations", min_value=0, max_value=10)  # Feature 32
        unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0, max_value=20.0, step=0.1)  # Feature 33
        inflation_rate = st.number_input("Inflation Rate", min_value=-10.0, max_value=10.0, step=0.1)  # Feature 34
        gdp = st.number_input("GDP", min_value=-10.0, max_value=10.0, step=0.1)  # Feature 35

    # Collect all the features into a list
    inputs = [marital_status, app_mode, app_order, course, attendance, prev_qualification,
              prev_qualification_grade, nationality, mother_qualification, father_qualification,
              mother_occupation, father_occupation, admission_grade, displaced, special_needs,
              debtor, tuition_fees, gender, scholarship, age, international, sem1_credited,
              sem1_enrolled, sem1_eval, sem1_approved, sem1_grade, sem1_without_eval, sem2_credited,
              sem2_enrolled, sem2_eval, sem2_approved, sem2_grade, sem2_without_eval, unemployment_rate,
              inflation_rate, gdp]

    # When user clicks the "Predict" button
    if st.button("Predict"):
        # Create input array (ensure it matches the feature order used in training, including the ID column)
        input_data = [student_id] + inputs  # Include the ID as the first feature
        
        # Predict student status
        try:
            status = predict_student_status(input_data)
            st.success(f"Student ID: {student_id}")
            st.write(f"The predicted student status is: **{status}**")
        except ValueError as e:
            st.error(f"Error: {str(e)}")

# --- CSV Upload Section ---
elif option == "Upload CSV":
    st.header("📁 Upload Student Data (CSV)")

    if 'features' not in st.session_state:
        st.session_state.features = None

    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file:
        # Load the uploaded CSV file
        uploaded_data = pd.read_csv(uploaded_file)
        
        # Ensure that the CSV file has the correct structure
        required_columns = ['id', 'Marital status', 'Application mode', 'Application order', 'Course',
                            'Daytime/evening attendance', 'Previous qualification', 
                            'Previous qualification (grade)', 'Nacionality', 
                            'Mother\'s qualification', 'Father\'s qualification',
                            'Mother\'s occupation', 'Father\'s occupation', 'Admission grade',
                            'Displaced', 'Educational special needs', 'Debtor', 
                            'Tuition fees up to date', 'Gender', 'Scholarship holder', 
                            'Age at enrollment', 'International',
                            'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 
                            'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 
                            'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 
                            'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
                            'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
                            'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 
                            'Unemployment rate', 'Inflation rate', 'GDP']

        # Check if the uploaded file has the required columns
        if set(uploaded_data.columns) == set(required_columns):
            # Extract the features from the first row of the uploaded CSV (excluding the ID)
            student_id = uploaded_data['id'].iloc[0]
            features = uploaded_data.drop(columns=['id']).values.flatten()
            
            # Store the features in session state
            st.session_state.features = dict(zip(uploaded_data.drop(columns=['id']).columns, features))

            # Get the prediction for the student
            try:
                status = predict_student_status([student_id] + features.tolist())
                st.success(f"Student ID: {student_id}")
                st.write(f"The predicted status is: **{status}**")
            except ValueError as e:
                st.error(f"Error: {str(e)}")
            
            # Button to display the features
            if st.button("See Values"):
                # Convert the features and values into a readable format
                st.write("**Features and Values**")
                for feature, value in st.session_state.features.items():
                    st.write(f"{feature}: {value}")
        else:
            st.error("The uploaded CSV file does not have the expected structure.")
