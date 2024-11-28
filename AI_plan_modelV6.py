import pandas as pd
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.callbacks import EarlyStopping
from flask import Flask, request, jsonify
import json
import os
import matplotlib.pyplot as plt

# File to store memory
MEMORY_FILE = "memory.json"

# Load memory from file
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as file:
            return json.load(file)
    return {}

# Save memory to file
def save_memory(memory):
    with open(MEMORY_FILE, 'w') as file:
        json.dump(memory, file)

# Initialize memory
memory = load_memory()

# Load and preprocess the dataset
gym_data = pd.read_csv('gym_exercise_dataset_V.2.csv')

# One-hot encode the dataset
categorical_columns = ['Equipment', 'Mechanics', 'Force', 'Main_muscle']
gym_data_encoded = pd.get_dummies(gym_data, columns=categorical_columns, drop_first=True)

# Define features (X) and labels (y)
X = gym_data_encoded.drop(columns=['Exercise Name', 'Difficulty (1-5)'])
y = pd.get_dummies(gym_data['Exercise Name'])  # Multi-label output: one-hot encoded exercise names

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the AI model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='sigmoid')  # Sigmoid for multi-label probabilities
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Helper functions
def filter_exercises(data, profile):
    filtered_data = data[
        (data["Difficulty (1-5)"] >= profile["difficulty_range"][0]) &
        (data["Difficulty (1-5)"] <= profile["difficulty_range"][1])
    ]
    filtered_data = filtered_data[
        filtered_data.apply(lambda row: any(eq in row["Equipment"] for eq in profile["equipment"]), axis=1)
    ]
    if profile["injuries"]:
        for injury in profile["injuries"]:
            filtered_data = filtered_data[filtered_data["Main_muscle"] != injury.capitalize()]
    return filtered_data

def prepare_filtered_data(filtered_data, original_columns, categorical_columns):
    encoded_filtered = pd.get_dummies(filtered_data, columns=categorical_columns, drop_first=True)
    encoded_filtered = encoded_filtered.reindex(columns=original_columns, fill_value=0)
    return encoded_filtered.astype('float32')

def rank_exercises_with_memory(filtered_data, model, original_columns, memory, user_key, recommended_count=6, compound_count=2):
    if filtered_data.empty:
        return pd.DataFrame(columns=['Exercise Name', 'Suitability', 'Main_muscle'])

    encoded_filtered = prepare_filtered_data(filtered_data, original_columns, ['Equipment', 'Mechanics', 'Force', 'Main_muscle'])
    predictions = model.predict(encoded_filtered)
    filtered_data = filtered_data.copy()
    filtered_data['Suitability'] = predictions.mean(axis=1)

    if user_key in memory:
        for exercise_name, row in filtered_data.iterrows():
            if exercise_name in memory[user_key]:
                frequency = memory[user_key][exercise_name]
                penalty = 0.1 * frequency
                filtered_data.at[exercise_name, 'Suitability'] *= (1 - penalty)

    filtered_data = filtered_data.sort_values(by='Suitability', ascending=False)

    compound_exercises = filtered_data[filtered_data['Mechanics'] == 'Compound']
    isolation_exercises = filtered_data[filtered_data['Mechanics'] != 'Compound']

    muscle_groups_seen = set()
    selected_compound_exercises = []

    for _, row in compound_exercises.iterrows():
        muscle_group = row['Main_muscle']
        if muscle_group not in muscle_groups_seen:
            selected_compound_exercises.append(row)
            muscle_groups_seen.add(muscle_group)
        if len(selected_compound_exercises) >= compound_count:
            break

    muscle_groups_seen.update([row['Main_muscle'] for row in selected_compound_exercises])
    final_recommendations = selected_compound_exercises

    for _, row in isolation_exercises.iterrows():
        muscle_group = row['Main_muscle']
        if muscle_group not in muscle_groups_seen:
            final_recommendations.append(row)
            muscle_groups_seen.add(muscle_group)
        if len(final_recommendations) >= recommended_count:
            break

    final_recommendations_df = pd.DataFrame(final_recommendations, columns=filtered_data.columns)

    if user_key not in memory:
        memory[user_key] = {}
    for exercise_name in final_recommendations_df['Exercise Name']:
        if exercise_name in memory[user_key]:
            memory[user_key][exercise_name] += 1
        else:
            memory[user_key][exercise_name] = 1

    return final_recommendations_df

# Flask API setup
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_profile = request.json

    # Provide default values for missing keys
    difficulty_range = user_profile.get("difficulty_range", [1, 5])
    equipment = user_profile.get("equipment", [])
    injuries = user_profile.get("injuries", [])

    # Ensure the profile has all necessary keys
    user_profile = {
        "difficulty_range": difficulty_range,
        "equipment": equipment,
        "injuries": injuries
    }

    filtered_exercises = filter_exercises(gym_data, user_profile)
    ranked_exercises = rank_exercises_with_memory(
        filtered_exercises, model, X.columns, memory,
        user_key="user_profile_key"
    )

    if ranked_exercises.empty:
        return jsonify({"message": "No recommendations could be generated."})
    else:
        return ranked_exercises[['Exercise Name', 'Main_muscle', 'Mechanics', 'Suitability']].to_json(orient='records')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
