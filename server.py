from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load Data and Train Model
def load_and_train_model():
    data_path = 'dataset/K4train.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return None, None

    df = pd.read_csv(data_path)
    
    # Preprocessing
    df = df[df['Age'] != 0]
    
    if 'Education_Level' in df.columns:
         mode_val = df['Education_Level'].mode()[0]
         df['Education_Level'].fillna(mode_val, inplace=True)
    
    # Define categorical columns to encode
    # Removed 'Per_Serving' and 'Total_Monthly' to treat them as numerical
    var_mod = ['Gender','Location','Occupation','Education_Level','Factor_Influence','Often_Restaurant','Preference']
    
    encoders = {}
    for col in var_mod:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(['Timestamp', 'Preference'], axis=1)
    y = df['Preference']
    
    model = KNeighborsClassifier(n_neighbors=7, p=1, weights='distance')
    model.fit(X.values, y.values)
    
    return model, encoders

model, encoders = load_and_train_model()

@app.route('/')
def index():
    # Pass options to frontend
    options = {}
    if encoders:
        for col, le in encoders.items():
            if col != 'Preference':
                options[col] = le.classes_.tolist()
    return render_template('index.html', options=options)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.json
    
    try:
        encoded_input = []
        encoded_input.append(int(data['Age']))
        encoded_input.append(encoders['Gender'].transform([data['Gender']])[0])
        encoded_input.append(encoders['Location'].transform([data['Location']])[0])
        encoded_input.append(encoders['Occupation'].transform([data['Occupation']])[0])
        encoded_input.append(encoders['Education_Level'].transform([data['Education_Level']])[0])
        
        # Use raw numerical values
        encoded_input.append(float(data['Per_Serving']))
        encoded_input.append(float(data['Total_Monthly']))
        
        encoded_input.append(encoders['Factor_Influence'].transform([data['Factor_Influence']])[0])
        encoded_input.append(encoders['Often_Restaurant'].transform([data['Often_Restaurant']])[0])
        encoded_input.append(float(data['Recommend_Traditional']))
        encoded_input.append(float(data['Recommend_International']))
        
        prediction_idx = model.predict([encoded_input])[0]
        prediction_label = encoders['Preference'].inverse_transform([prediction_idx])[0]
        
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
