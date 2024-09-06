
import streamlit as st
import pandas as pd
import psycopg2
import sqlalchemy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sqlalchemy import create_engine

# Establish the database connection
try:
    engine = create_engine("postgresql://postgres:1q2w3e4@localhost:5432/telecom")
    st.write("Database connected successfully!")

except Exception as e:
    st.write(f"Error: {e}")

# Query to fetch the data
query = """
SELECT * FROM xdr_data;
"""

# Fetch the data into a Pandas DataFrame
df = pd.read_sql(query, engine)

# Check for required columns
required_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                    'Total Data (DL+UL)']

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.write(f"Missing columns: {missing_columns}")
else:
    # Perform checks on the data
    st.write("Data fetched from database:")
    st.write(df.head())

    st.write("Data types:")
    st.write(df.dtypes)

    # Convert relevant columns to numeric, forcing errors to NaN
    numeric_columns = df.select_dtypes(include=['object']).columns  # Identify object-type columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, set errors to NaN

    # Fill NaN values if necessary
    df.fillna(df.mean(), inplace=True)  # This will work now if numeric types are correct

    st.write("Filled NaN values:")
    st.write(df.head())

    # Handle Outliers by replacing them with the mean for values with z-score > 3
    for column in numeric_columns:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df[column] = np.where(z_scores > 3, df[column].mean(), df[column])

    st.write("Handled outliers:")
    st.write(df.head())

    # Segment Users into Top 5 Decile Classes based on Total Session Duration
    df['Total Data (DL+UL)'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    df['Duration Decile'] = pd.qcut(df['Dur. (ms)'], 5, labels=False)

    # Compute total data per decile class
    decile_data = df.groupby('Duration Decile')['Total Data (DL+UL)'].sum()
    st.write(decile_data)

    # Basic Metrics (Mean, Median, etc.)
    st.write(df.mean())  # Mean of each column
    st.write(df.median())  # Median of each column

    # Non-Graphical Univariate Analysis (Dispersion Measures)
    st.write(df.var())  # Variance
    st.write(df.std())  # Standard Deviation
    st.write(df.skew())  # Skewness
    st.write(df.columns)  # Print the columns in the DataFrame
    df.columns = df.columns.str.strip()  # Remove leading and trailing spaces
    st.write("Columns in the DataFrame:")
    st.write(df.columns.tolist())  

    # Graphical Univariate Analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Total Data (DL+UL)'], bins=30, kde=True)
    plt.title('Distribution of Total Data (DL+UL)')
    st.pyplot()

    # Bivariate Analysis (Relationship between Applications and Total DL+UL Data)
    sns.pairplot(df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                     'Total Data (DL+UL)']])
    st.pyplot()

    # Correlation Analysis
    correlation_matrix = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                             'Total Data (DL+UL)']].corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix for Application Data')
    st.pyplot()

    # Dimensionality Reduction - PCA
    features = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Total Data (DL+UL)']
    X = df[features].fillna(0)  # Handle missing values again if necessary
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions
    principal_components = pca.fit_transform(X_scaled)

    st.write("Principal Components:")
    st.write(principal_components)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=df['Duration Decile'], cmap='viridis')
    plt.colorbar(label='Decile Class')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA on Application Data')
    st.pyplot()

    # Explained Variance
    st.write("Explained variance ratio:", pca.explained_variance_ratio_)



