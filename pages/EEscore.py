import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import streamlit as st

# Streamlit app setup
st.title("Customer Experience and Engagement Score")

# Load user data
user_data = pd.read_csv('./notebooks/Week2_data.csv')
st.write("Columns in user_data:", user_data.columns)

# Engagement and Experience clusters
engagement_clusters = {
    'less_engaged': np.array([20, 30]),  # Example values for engagement cluster
}

experience_clusters = {
    'worst_experience': np.array([5]),  # Example values for experience cluster
}

# Function to calculate Euclidean distance
def calculate_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

# Function to assign scores
def assign_scores(user_data, engagement_clusters, experience_clusters):
    engagement_scores = []
    experience_scores = []
    
    # Centroids from clusters
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

    return user_data

# Assign scores to each user
scored_user_data = assign_scores(user_data, engagement_clusters, experience_clusters)

# Display results on Streamlit dashboard
st.header("Calculated Scores")
st.dataframe(scored_user_data)

# Visualize results (e.g., scatter plot, bar chart)
st.subheader("Engagement Score Distribution")
st.bar_chart(scored_user_data['engagement_score'])

st.subheader("Experience Score Distribution")
st.bar_chart(scored_user_data['experience_score'])

# Step 1: Save the scored data to CSV
scored_user_data.to_csv('scored_user_data.csv', index=False)

# Step 2: Provide a download button in Streamlit
st.download_button(
    label="Download Scored Data as CSV",
    data=scored_user_data.to_csv(index=False).encode('utf-8'),
    file_name='scored_user_data.csv',
    mime='text/csv'
)
