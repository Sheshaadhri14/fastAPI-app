from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['EngineType', 'EnergySource'])

# Convert RecordID to string
data['RecordID'] = data['RecordID'].astype(str)

# Select only numeric columns for the anomaly detection
X_numeric = data.drop(['RecordID', 'AmountConsumed'], axis=1).select_dtypes(include=['number']).astype(float)

# Calculate similarity matrix based on location and engine type
def calculate_similarity_matrix(data):
    location_columns = [col for col in data.columns if col.startswith('Location_')]
    engine_type_columns = [col for col in data.columns if col.startswith('EngineType_')]
    features = data[location_columns + engine_type_columns]
    similarity_matrix = cosine_similarity(features, features)
    return similarity_matrix

similarity_matrix = calculate_similarity_matrix(data)

# Detect anomalies
anomaly_mask = anomaly_model.predict(X_numeric) == -1
anomaly_records = data[anomaly_mask]

# Function to recommend solutions
def recommend_solutions(record_id, similarity_matrix, data):
    record_index = data[data['RecordID'] == record_id].index[0]
    best_solution = data.iloc[record_index].to_dict()
    return best_solution

class RecordID(BaseModel):
    record_id: str

@app.get('/detect_anomalies', response_model=list[str])
def detect_anomalies():
    anomalies = anomaly_records['RecordID'].tolist()
    return anomalies

@app.post('/recommend_solution', response_model=dict)
def recommend_solution(record: RecordID):
    record_id = record.record_id
    if record_id not in data['RecordID'].values:
        raise HTTPException(status_code=404, detail="Record ID not found")
    best_solution = recommend_solutions(record_id, similarity_matrix, data)
    response = {
        'RecordID': str(best_solution['RecordID']),
    }
    return response

@app.get('/recommend_solutions_for_anomalies', response_model=list[dict])
def recommend_solutions_for_anomalies():
    solutions = []
    for record_id in anomaly_records['RecordID']:
        best_solution = recommend_solutions(record_id, similarity_matrix, data)
        solution = {
            'AnomalyRecordID': record_id,
            'SolutionRecordID': str(best_solution['RecordID']),
        }
        solutions.append(solution)
    return solutions
