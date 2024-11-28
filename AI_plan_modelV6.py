from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import json
import os

# Load and preprocess the dataset
gym_data = pd.read_csv('gym_exercise_dataset_V.2.csv')

# Preprocess the dataset
categorical_columns = ['Equipment', 'Mechanics', 'Force', 'Main_muscle']
gym_data_encoded = pd.get_dummies(gym_data, columns=categorical_columns, drop_first=True)

# Define features (X) and labels (y)
X = gym_data_encoded.drop(columns=['Exercise Name', 'Difficulty (1-5)'])
y = pd.get_dummies(gym_data['Exercise Name'])  # Multi-label output: one-hot encoded exercise names

# Load the pretrained AI model
model = load_model('gym_exercise_model.h5')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 1: Filter dataset based on user constraints
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


# Step 2: Prepare filtered data for prediction
def prepare_filtered_data(filtered_data, original_columns, categorical_columns):
    # One-hot encode filtered data
    encoded_filtered = pd.get_dummies(filtered_data, columns=categorical_columns, drop_first=True)
    # Align columns with training data
    encoded_filtered = encoded_filtered.reindex(columns=original_columns, fill_value=0)
    return encoded_filtered.astype('float32')


# Step 3: Rank exercises using the model
def rank_exercises(filtered_data, model, original_columns, recommended_count=6, compound_count=2):
    if filtered_data.empty:
        return pd.DataFrame(columns=['Exercise Name', 'Suitability', 'Main_muscle'])

    # Prepare the filtered dataset
    encoded_filtered = prepare_filtered_data(filtered_data, original_columns,
                                             ['Equipment', 'Mechanics', 'Force', 'Main_muscle'])

    # Predict suitability
    predictions = model.predict(encoded_filtered)

    # Add suitability scores
    filtered_data = filtered_data.copy()
    filtered_data['Suitability'] = predictions.mean(axis=1)

    # Sort by Suitability
    filtered_data = filtered_data.sort_values(by='Suitability', ascending=False)

    # Separate compound and isolation exercises
    compound_exercises = filtered_data[filtered_data['Mechanics'] == 'Compound']
    isolation_exercises = filtered_data[filtered_data['Mechanics'] != 'Compound']

    # Select top diverse compound exercises
    muscle_groups_seen = set()
    selected_compound_exercises = []

    for _, row in compound_exercises.iterrows():
        muscle_group = row['Main_muscle']
        if muscle_group not in muscle_groups_seen:
            selected_compound_exercises.append(row)
            muscle_groups_seen.add(muscle_group)
        if len(selected_compound_exercises) >= compound_count:
            break

    # Fill the rest with diverse isolation exercises
    muscle_groups_seen.update([row['Main_muscle'] for row in selected_compound_exercises])
    final_recommendations = selected_compound_exercises

    for _, row in isolation_exercises.iterrows():
        muscle_group = row['Main_muscle']
        if muscle_group not in muscle_groups_seen:
            final_recommendations.append(row)
            muscle_groups_seen.add(muscle_group)
        if len(final_recommendations) >= recommended_count:
            break

    # Convert back to DataFrame and assign proper column names
    final_recommendations_df = pd.DataFrame(final_recommendations, columns=filtered_data.columns)

    return final_recommendations_df


# Flask app setup
app = Flask(__name__)


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get the user profile from the request
        user_profile = request.json

        # Filter exercises based on the user profile
        filtered_exercises = filter_exercises(gym_data, user_profile)

        # Rank exercises using the pretrained model
        ranked_exercises = rank_exercises(filtered_exercises, model, X.columns, recommended_count=6, compound_count=2)

        # If no exercises are found
        if ranked_exercises.empty:
            return jsonify({"message": "No recommendations could be generated."})

        # Return the ranked exercises as JSON
        return ranked_exercises[['Exercise Name', 'Main_muscle', 'Mechanics', 'Suitability']].to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
