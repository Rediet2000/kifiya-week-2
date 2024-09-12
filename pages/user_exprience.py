import pandas as pd
import numpy as np
from scipy.stats import zscore
import streamlit as st

# Function to display content for Clustering Analysis
# def display():
#     #st.title("Clustering Analysis")
#     st.write("This is the Clustering Analysis page.")
#     # Add relevant components here, like:
#     st.bar_chart([3, 5, 7, 9, 11, 13, 15])


df = pd.read_csv('./notebooks/Week2_data.csv')

df.head()


# Cell 2: Handle missing values in numerical columns
df['TCP DL Retrans. Vol (Bytes)'] = df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean())
df['TCP UL Retrans. Vol (Bytes)'] = df['TCP UL Retrans. Vol (Bytes)'].fillna(df['TCP UL Retrans. Vol (Bytes)'].mean())
df['Avg RTT DL (ms)'] = df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean())
df['Avg RTT UL (ms)'] = df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean())
df['Avg Bearer TP DL (kbps)'] = df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean())
df['Avg Bearer TP UL (kbps)'] = df['Avg Bearer TP UL (kbps)'].fillna(df['Avg Bearer TP UL (kbps)'].mean())
df['Handset Type'] = df['Handset Type'].fillna(df['Handset Type'].mode()[0])

# Display the DataFrame to verify that missing values are handled
df.head()


def handle_outliers(column):
    column_data = df[column].dropna()
    
    z_scores = zscore(column_data)
    
    abs_z_scores = np.abs(z_scores)
    outlier_indices = np.where(abs_z_scores > 3)
    
    df.loc[df.index[outlier_indices[0]], column] = df[column].mean()

handle_outliers('TCP DL Retrans. Vol (Bytes)')
handle_outliers('TCP UL Retrans. Vol (Bytes)')
handle_outliers('Avg RTT DL (ms)')
handle_outliers('Avg RTT UL (ms)')
handle_outliers('Avg Bearer TP DL (kbps)')
handle_outliers('Avg Bearer TP UL (kbps)')

df.head()