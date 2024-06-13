from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from joblib import load
import pickle

app = FastAPI()

# Load pre-trained models and feature names
anomaly_model = load('anomaly_model.joblib')
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Read the Excel file into a DataFrame (assuming it's stored in the same directory)
data = pd.read_excel("data/ENERGY.xlsx")

# Drop irrelevant columns and datetime columns
data.drop(['UserID', 'Timestamp'], axis=1, inplace=True)

# Handle missing values
data.dropna(inplace=True)

# Convert RecordID to string
data['RecordID'] = data['RecordID'].astype(str)

# Select only numeric columns for the anomaly detection
X_numeric = data.drop(['RecordID', 'AmountConsumed'], axis=1).select_dtypes(include=['number']).astype(float)

# Function to recommend solutions
def recommend_solutions(record_id, data):
    record_index = data[data['RecordID'] == record_id].index[0]
    return {'RecordID': str(data.loc[record_index, 'RecordID'])}

class RecordID(BaseModel):
    record_id: str

@app.get('/detect_anomalies', response_model=list[str])
def detect_anomalies():
    anomalies = data['RecordID'].tolist()
    return anomalies

@app.post('/recommend_solution', response_model=dict)
def recommend_solution(record: RecordID):
    record_id = record.record_id
    if record_id not in data['RecordID'].values:
        raise HTTPException(status_code=404, detail="Record ID not found")
    best_solution = recommend_solutions(record_id, data)
    response = {
        'RecordID': best_solution['RecordID'],
    }
    return response

@app.get('/recommend_solutions_for_anomalies', response_model=list[dict])
def recommend_solutions_for_anomalies():
    solutions = []
    for record_id in anomaly_records['RecordID']:
        best_solution = recommend_solutions(record_id, similarity_matrix, data)
        solution = {
            'RecordID': best_solution['RecordID'],
        }
        solutions.append(solution)
    return solutions

