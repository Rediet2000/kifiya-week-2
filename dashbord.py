import streamlit as st
import pandas as pd
import psycopg2

# Set up database connection
def get_db_connection():
    connection = psycopg2.connect(
        host='localhost',  # replace with your database host
        database='telecom',  # replace with your database name
        user='postgres',  # replace with your database username
        password='1q2w3e4r'  # replace with your database password
    )
    return connection

# Query user behavior data
def query_user_behavior():
    connection = get_db_connection()
    query = """
        SELECT 
            "MSISDN/Number" AS user_id,
            COUNT(*) AS number_of_sessions,
            SUM("Dur. (ms)") AS total_session_duration,
            SUM("Total DL (Bytes)") AS total_download_data,
            SUM("Total UL (Bytes)") AS total_upload_data
        FROM 
            public.xdr_data
        GROUP BY 
            "MSISDN/Number";
    """
    df = pd.read_sql(query, connection)
    connection.close()
    return df

# Streamlit app
def main():
    st.title("User Behavior Dashboard")
    
    st.header("User Behavior Overview")
    
    df = query_user_behavior()
    
    if not df.empty:
        st.dataframe(df)
        
        # Plotting session duration and data usage
        st.subheader("Session Duration and Data Usage")
        st.bar_chart(df[['user_id', 'total_download_data', 'total_upload_data']].set_index('user_id'))

    else:
        st.write("No data available.")

if __name__ == "__main__":
    main()
