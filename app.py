import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Set page config
st.set_page_config(page_title="Food Preference Predictor", layout="wide")

st.title("Food Preference Prediction System")
st.write("This system predicts food preference (Local vs International) based on user demographics and eating habits using K-Nearest Neighbors.")

# Load Data
@st.cache_data
def load_and_train_model():
    data_path = 'dataset/K4train.csv'
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}")
        return None, None, None

    df = pd.read_csv(data_path)
    
    # Preprocessing
    # Drop rows with Age = 0 (as per notebook)
    df = df[df['Age'] != 0]
    
    # Fill missing Education_Level with mode
    if 'Education_Level' in df.columns:
         mode_val = df['Education_Level'].mode()[0]
         df['Education_Level'].fillna(mode_val, inplace=True)
    
    # Define categorical columns to encode
    var_mod = ['Gender','Location','Occupation','Education_Level','Per_Serving','Total_Monthly','Factor_Influence','Often_Restaurant','Preference']
    
    encoders = {}
    for col in var_mod:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Prepare X and y
    # Drop Timestamp and Preference from X
    X = df.drop(['Timestamp', 'Preference'], axis=1)
    y = df['Preference']
    
    # Train Model
    model = KNeighborsClassifier(n_neighbors=7, p=1, weights='distance')
    model.fit(X.values, y.values)
    
    return model, encoders, df

model, encoders, df_train = load_and_train_model()

if model is None:
    st.stop()

# User Input Form
st.sidebar.header("User Input Features")

def user_input_features():
    # We need the original unique values for selectboxes. 
    # Since we encoded df_train, we can use encoders[col].classes_
    
    # Age (Numerical)
    age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25)
    
    # Gender
    gender_options = encoders['Gender'].classes_
    gender = st.sidebar.selectbox("Gender", gender_options)
    
    # Location
    location_options = encoders['Location'].classes_
    location = st.sidebar.selectbox("Location", location_options)
    
    # Occupation
    occupation_options = encoders['Occupation'].classes_
    occupation = st.sidebar.selectbox("Occupation", occupation_options)
    
    # Education_Level
    edu_options = encoders['Education_Level'].classes_
    education = st.sidebar.selectbox("Education Level", edu_options)
    
    # Per_Serving (Encoded in notebook, so treated as categorical/discrete)
    # It's numerical in nature, but we must stick to training values or handle new ones.
    # Using selectbox ensures we pick from known values.
    per_serving_options = encoders['Per_Serving'].classes_
    per_serving = st.sidebar.selectbox("Cost Per Serving", per_serving_options)
    
    # Total_Monthly
    total_monthly_options = encoders['Total_Monthly'].classes_
    total_monthly = st.sidebar.selectbox("Total Monthly Spending on Food", total_monthly_options)
    
    # Factor_Influence
    factor_options = encoders['Factor_Influence'].classes_
    factor = st.sidebar.selectbox("Factor Influencing Choice", factor_options)
    
    # Often_Restaurant
    often_options = encoders['Often_Restaurant'].classes_
    often = st.sidebar.selectbox("How Often Do You Eat at Restaurants?", often_options)
    
    # Recommend_Traditional (Numerical)
    rec_trad = st.sidebar.slider("Likelihood to Recommend Traditional Food (1-10)", 1.0, 10.0, 5.0)
    
    # Recommend_International (Numerical)
    rec_inter = st.sidebar.slider("Likelihood to Recommend International Food (1-10)", 1.0, 10.0, 5.0)
    
    data = {
        'Age': age,
        'Gender': gender,
        'Location': location,
        'Occupation': occupation,
        'Education_Level': education,
        'Per_Serving': per_serving,
        'Total_Monthly': total_monthly,
        'Factor_Influence': factor,
        'Often_Restaurant': often,
        'Recommend_Traditional': rec_trad,
        'Recommend_International': rec_inter
    }
    return data

input_data = user_input_features()

# Display User Input
st.subheader("User Input parameters")
st.write(pd.DataFrame([input_data]))

# Prediction Button
if st.button("Predict Preference"):
    # Encode inputs
    encoded_input = []
    
    # Order must match X columns: 
    # Age, Gender, Location, Occupation, Education_Level, Per_Serving, Total_Monthly, Factor_Influence, Often_Restaurant, Recommend_Traditional, Recommend_International
    
    # Age
    encoded_input.append(input_data['Age'])
    
    # Gender
    encoded_input.append(encoders['Gender'].transform([input_data['Gender']])[0])
    
    # Location
    encoded_input.append(encoders['Location'].transform([input_data['Location']])[0])
    
    # Occupation
    encoded_input.append(encoders['Occupation'].transform([input_data['Occupation']])[0])
    
    # Education_Level
    encoded_input.append(encoders['Education_Level'].transform([input_data['Education_Level']])[0])
    
    # Per_Serving
    encoded_input.append(encoders['Per_Serving'].transform([input_data['Per_Serving']])[0])
    
    # Total_Monthly
    encoded_input.append(encoders['Total_Monthly'].transform([input_data['Total_Monthly']])[0])
    
    # Factor_Influence
    encoded_input.append(encoders['Factor_Influence'].transform([input_data['Factor_Influence']])[0])
    
    # Often_Restaurant
    encoded_input.append(encoders['Often_Restaurant'].transform([input_data['Often_Restaurant']])[0])
    
    # Recommend_Traditional
    encoded_input.append(input_data['Recommend_Traditional'])
    
    # Recommend_International
    encoded_input.append(input_data['Recommend_International'])
    
    # Predict
    prediction_idx = model.predict([encoded_input])[0]
    prediction_label = encoders['Preference'].inverse_transform([prediction_idx])[0]
    
    st.subheader("Prediction")
    st.success(f"The predicted food preference is: **{prediction_label}**")

