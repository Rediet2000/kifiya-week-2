import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import streamlit as st

# Streamlit app setup
st.title("Customer Experience and Engagement Score")

# Load user data
user_data = pd.read_csv('./notebooks/Week2_data.csv')
st.write("Columns in user_data:", user_data.columns)

# Define engagement and experience clusters
engagement_clusters = {
    'less_engaged': np.array([20, 30]),  # Example values for engagement cluster
}

experience_clusters = {
    'worst_experience': np.array([5]),  # Example values for experience cluster
}

# Function to calculate Euclidean distance
def calculate_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

# Calculate engagement and experience scores
def assign_scores(user_data, engagement_clusters, experience_clusters):
    engagement_scores = []
    experience_scores = []
    
    # Extract centroids from clusters
    less_engaged_centroid = engagement_clusters['less_engaged']
    worst_experience_centroid = experience_clusters['worst_experience']
    
    for index, row in user_data.iterrows():
        # Adjust column names based on your CSV file
        user_point = np.array([row['Avg Bearer TP DL (kbps)'], row['Avg Bearer TP UL (kbps)']])
        
        # Calculate engagement score
        engagement_score = calculate_euclidean_distance(user_point, less_engaged_centroid)
        engagement_scores.append(engagement_score)
        
        # Calculate experience score
        experience_score = calculate_euclidean_distance(np.array([row['DL TP < 50 Kbps (%)']]), worst_experience_centroid)
        experience_scores.append(experience_score)
    
    user_data['engagement_score'] = engagement_scores
    user_data['experience_score'] = experience_scores

    # Calculate satisfaction score as the average of engagement and experience scores
    user_data['satisfaction_score'] = (user_data['engagement_score'] + user_data['experience_score']) / 2

    return user_data

# Assign scores to each user
scored_user_data = assign_scores(user_data, engagement_clusters, experience_clusters)

# Report the top 10 most satisfied customers
top_10_satisfied_customers = scored_user_data.nlargest(10, 'satisfaction_score')

# Print the top 10 satisfied customers
st.write("Top 10 Satisfied Customers:")
st.dataframe(top_10_satisfied_customers[['IMEI', 'satisfaction_score']])  # Adjust 'IMEI' as needed

# Step 1: Save the scored data to CSV
scored_user_data.to_csv('scored_user_data.csv', index=False)

# Step 2: Provide a download button in Streamlit
st.download_button(
    label="Download Scored Data as CSV",
    data=scored_user_data.to_csv(index=False).encode('utf-8'),
    file_name='scored_user_data.csv',
    mime='text/csv'
)

# Optionally provide a download button for the top 10 satisfied customers
st.download_button(
    label="Download Top 10 Satisfied Customers as CSV",
    data=top_10_satisfied_customers.to_csv(index=False).encode('utf-8'),
    file_name='top_10_satisfied_customers.csv',
    mime='text/csv'
)
