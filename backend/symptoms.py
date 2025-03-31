import sys
import numpy as np
import pandas as pd
import pickle
import json

# Load Symptoms and Diseases Dictionaries
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12
}
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes',
    40: 'Hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C'
}

# Load Datasets
sym_des = pd.read_csv(r"E:\Projects\EMA\Effective-medical-assistant\backend\HealthPredict\symtoms_df.csv")
precautions = pd.read_csv(r"E:\Projects\EMA\Effective-medical-assistant\backend\HealthPredict\precautions_df.csv")
workout = pd.read_csv(r"E:\Projects\EMA\Effective-medical-assistant\backend\HealthPredict\workout_df.csv")
description = pd.read_csv(r"E:\Projects\EMA\Effective-medical-assistant\backend\HealthPredict\description.csv")
medications = pd.read_csv(r"E:\Projects\EMA\Effective-medical-assistant\backend\HealthPredict\medications.csv")
diets = pd.read_csv(r"E:\Projects\EMA\Effective-medical-assistant\backend\HealthPredict\diets.csv")

# Load Model
with open(r"E:\Projects\EMA\Effective-medical-assistant\backend\aimodels\svc.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

# Function to Predict Disease
def get_predicted_value(symptoms):
    symptoms = symptoms.split(',')  
    symptoms = [symptom.strip() for symptom in symptoms]  
    input_vector = np.zeros(len(symptoms_dict))
    warnings = []
    
    for symptom in symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            warnings.append(f"Symptom '{symptom}' not found in symptoms_dict")

    # Get Prediction
    predicted_label = int(model.predict([input_vector])[0])  # Convert np.int32 to int

    # Handle KeyError (If disease is not found)
    predicted_disease = diseases_list.get(predicted_label, "Unknown Disease")
    if predicted_disease == "Unknown Disease":
        warnings.append(f"Predicted label '{predicted_label}' not found in diseases_list")

    return predicted_disease, warnings

# Helper Function to Fetch Disease Details
def helper(dis):
    desc = description.loc[description['Disease'] == dis, 'Description']
    desc = " ".join(desc.tolist())

    pre = precautions.loc[precautions['Disease'] == dis, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.tolist()[0] if not pre.empty else []

    med = medications.loc[medications['Disease'] == dis, 'Medication']
    med = med.tolist() if not med.empty else []

    die = diets.loc[diets['Disease'] == dis, 'Diet']
    die = die.tolist() if not die.empty else []

    wrkout = workout.loc[workout['disease'] == dis, 'workout']
    wrkout = wrkout.tolist() if not wrkout.empty else []

    return desc, pre, med, die, wrkout

# Get Input from Command Line
data = sys.argv[3]
data_dict = json.loads(data)
symptoms = data_dict.get('data', '')

# Predict Disease and Handle Warnings
print("Received Symptoms:", symptoms)
predicted_disease, warnings = get_predicted_value(symptoms)

# Get Details
dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

# Prepare Final JSON Output
data_output = {
    "predicted_disease": str(predicted_disease),
    "dis_des": str(dis_des),
    "my_precautions": precautions,
    "medications": medications,
    "rec_diet": rec_diet,
    "workout": workout,
    "warnings": warnings
}

# Convert to JSON and Print
json_string = json.dumps(data_output, indent=2)
print(json_string)
