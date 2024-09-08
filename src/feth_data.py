import pandas as pd
import psycopg2

# Database configuration
db_config = {
    'dbname': 'telecom',
    'user': 'postgres',
    'password': '',
    'host': 'localhost',
    'port': '5432'
}

# Connect to the PostgreSQL database
def create_connection(config):
    return psycopg2.connect(**config)

# Fetch data from the database
def fetch_data(conn, query):
    return pd.read_sql_query(query, conn)

# Define your SQL query to fetch the dataset
query = """
SELECT *
FROM your_table_name
"""

# Create connection and fetch data
conn = create_connection(db_config)
df = fetch_data(conn, query)

# Close the connection
conn.close()

# Save the data to a CSV file for analysis
df.to_csv('data/telecom_data.csv', index=False)

print("Data has been fetched and saved to 'data/telecom_data.csv'.")
