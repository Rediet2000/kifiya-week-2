import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Title for the Streamlit dashboard
st.title('User Data Regression Model')

# Load user data from a CSV file
user_data = pd.read_csv('./data/scored_user_data.csv')

# Show data preview on Streamlit
st.write("### Data Preview")
st.dataframe(user_data.head())

# Example centroids for clusters (replace with your actual centroids)
engagement_clusters = {'less_engaged': np.array([20, 30])}
experience_clusters = {'worst_experience': np.array([5])}

# Function to calculate Euclidean distance
def calculate_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

# Function to assign scores
def assign_scores(user_data, engagement_clusters, experience_clusters):
    engagement_scores = []
    experience_scores = []

    # Extract centroids
    less_engaged_centroid = engagement_clusters['less_engaged']
    worst_experience_centroid = experience_clusters['worst_experience']

    for index, row in user_data.iterrows():
        user_point = np.array([row['Avg Bearer TP DL (kbps)'], row['Avg Bearer TP UL (kbps)']])

        engagement_score = calculate_euclidean_distance(user_point, less_engaged_centroid)
        engagement_scores.append(engagement_score)

        experience_score = calculate_euclidean_distance(np.array([row['DL TP < 50 Kbps (%)']]), worst_experience_centroid)
        experience_scores.append(experience_score)

    user_data['engagement_score'] = engagement_scores
    user_data['experience_score'] = experience_scores

    return user_data

# Assign scores to each user
scored_user_data = assign_scores(user_data, engagement_clusters, experience_clusters)

# Drop rows where 'satisfaction_score' is NaN
scored_user_data = scored_user_data.dropna(subset=['satisfaction_score'])

# Handling missing values (NaN) in X by imputing the mean
imputer = SimpleImputer(strategy='mean')

# Select the features (engagement_score, experience_score) and the target (satisfaction_score)
X = scored_user_data[['engagement_score', 'experience_score']]  # Features
y = scored_user_data['satisfaction_score']  # Target

# Impute missing values in X
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results on Streamlit
st.write("### Model Performance")
st.write(f"**Mean Squared Error:** {mse:.4f}")
st.write(f"**R-squared:** {r2:.4f}")

# Plot Actual vs Predicted values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted Satisfaction Scores')

# Show the plot in Streamlit
st.pyplot(fig)

# Show sample of the scored user data
st.write("### Scored User Data")
st.dataframe(scored_user_data[['engagement_score', 'experience_score', 'satisfaction_score']].head())
