import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# Option to drop or fill missing values
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
st.subheader("Handset Type Distribution")
handset_counts = df['Handset Type'].value_counts()
st.bar_chart(handset_counts)

# Comparing Handset Type with another feature, e.g., Total DL (Bytes)
st.subheader("Comparison: Handset Type vs. Total DL (Bytes)")
df_clean = df.dropna(subset=["Handset Type", "Total DL (Bytes)"])  # Clean data

fig, ax = plt.subplots()
sns.boxplot(x="Handset Type", y="Total DL (Bytes)", data=df_clean, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)
