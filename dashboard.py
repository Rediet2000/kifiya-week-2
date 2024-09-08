import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import psycopg2
from sqlalchemy import create_engine

# Establish connection
conn = psycopg2.connect(
    dbname="telecom",
    user="postgres",
    password="new_password",
    host="localhost",
    port="5432"
)

# Create a SQLAlchemy engine from psycopg2 connection
engine = create_engine('postgresql+psycopg2://postgres:new_password@localhost:5432/telecom')

# Load data from PostgreSQL using pandas
df = pd.read_sql_query("SELECT * FROM xdr_data", engine)

# Set page title
st.title("Telecom Data - Exploratory Data Analysis")

# Show first few rows of the dataset
st.subheader("Initial Data")
st.write(df.head())

# Handle Missing Values
st.subheader("Handle Missing Values")
missing_columns = df.columns[df.isnull().any()]
missing_column = st.selectbox("Choose a column to fill missing values", missing_columns)
fill_method = st.radio("Fill method", ["Fill with Mean", "Fill with Median", "Drop Rows"])

if st.button("Apply Fill"):
    if fill_method == "Fill with Mean":
        df[missing_column] = df[missing_column].fillna(df[missing_column].mean())
    elif fill_method == "Fill with Median":
        df[missing_column] = df[missing_column].fillna(df[missing_column].median())
    else:
        df = df.dropna(subset=[missing_column])
    st.success(f"{missing_column} cleaned successfully")

# ---- 2. Descriptive Statistics ----
st.subheader("Descriptive Statistics")
st.write(df.describe())

# ---- 3. Visualizations ----

# Handset Type Distribution
st.subheader("Top 10 Handsets")
top_10_handsets = df['handset'].value_counts().head(10)
st.bar_chart(top_10_handsets)

# Top 3 Handset Manufacturers
st.subheader("Top 3 Handset Manufacturers")
top_3_manufacturers = df['manufacturer'].value_counts().head(3)
st.bar_chart(top_3_manufacturers)

# Top 5 Handsets per Top 3 Manufacturers
st.subheader("Top 5 Handsets per Top 3 Manufacturers")
top_3_manufacturers_list = top_3_manufacturers.index
top_5_handsets_per_manufacturer = {}
for manufacturer in top_3_manufacturers_list:
    handsets = df[df['manufacturer'] == manufacturer]['handset'].value_counts().head(5)
    top_5_handsets_per_manufacturer[manufacturer] = handsets
for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
    st.write(f"Top 5 handsets for {manufacturer}:\n{handsets}")

# Visualize Handset Type vs Total DL (Bytes)
st.subheader("Comparison: Handset Type vs. Total DL (Bytes)")
df_clean = df.dropna(subset=["handset", "total_data"])  # Clean data
fig, ax = plt.subplots()
sns.boxplot(x="handset", y="total_data", data=df_clean, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# Univariate Analysis
st.subheader("Univariate Analysis")
st.write("Distribution plots for 'total_data', 'data_DL', and 'data_UL'")
fig, ax = plt.subplots(3, 1, figsize=(10, 15))
sns.histplot(df['total_data'], kde=True, ax=ax[0])
ax[0].set_title('Total Data Distribution')
sns.histplot(df['data_DL'], kde=True, ax=ax[1])
ax[1].set_title('Download Data Distribution')
sns.histplot(df['data_UL'], kde=True, ax=ax[2])
ax[2].set_title('Upload Data Distribution')
st.pyplot(fig)

# Bivariate Analysis
st.subheader("Bivariate Analysis")
applications = ['Social_Media', 'Google', 'Email', 'YouTube', 'Netflix', 'Gaming', 'Others']
for app in applications:
    st.subheader(f"{app} vs Total Data Usage")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=app, y='total_data', ax=ax)
    ax.set_title(f"{app} vs Total Data Usage")
    st.pyplot(fig)

# Correlation Analysis
st.subheader("Correlation Analysis")
correlation_matrix = df[applications].corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Dimensionality Reduction - PCA
st.subheader("Dimensionality Reduction - PCA")
features = df[applications]
scaled_features = StandardScaler().fit_transform(features)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
st.write(pca_df.head())
st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

fig, ax = plt.subplots()
sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax)
ax.set_title('PCA - Principal Components')
st.pyplot(fig)
