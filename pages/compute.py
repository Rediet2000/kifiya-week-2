import numpy as np
import pandas as pd
import streamlit as st
    
def display():
    #st.title("Throughput Analysis")
    #st.write("This is the Throughput Analysis page.")
    st.line_chart([1, 2, 3, 4, 5,6,7,8,9,10,11,12])

data = pd.read_csv('./notebooks/Week2_data.csv')
data.head()

data['Throughput'] = (data['Total DL (Bytes)'] + data['Total UL (Bytes)']) / data['Dur. (ms)']
top_throughput = data['Throughput'].nlargest(10)
bottom_throughput = data['Throughput'].nsmallest(10)
most_frequent_throughput = data['Throughput'].value_counts().nlargest(10)
print("Top 10 Throughput Values:\n", top_throughput)
print("Bottom 10 Throughput Values:\n", bottom_throughput)
print("Most Frequent 10 Throughput Values:\n", most_frequent_throughput)

data['Start'] = pd.to_datetime(data['Start'])
data['End'] = pd.to_datetime(data['End'])
data['RTT'] = (data['End'] - data['Start']).dt.total_seconds() * 1000  # Convert seconds to milliseconds


average_rtt = data['RTT'].mean()
print(f'Average RTT: {average_rtt}')

if 'Packet_Loss' in data.columns:
    average_packet_size = 1500  

    data['Total_Packets_Sent'] = (data['Total DL (Bytes)'] + data['Total UL (Bytes)']) / average_packet_size
    data['Estimated_Retransmissions'] = data['Total_Packets_Sent'] * (data['Packet_Loss'] / 100)

    average_retransmission = data['Estimated_Retransmissions'].mean()

    print(f'Average Estimated TCP Retransmission Rate: {average_retransmission}')
else:
    print('Packet_Loss column is missing in the dataset.')

