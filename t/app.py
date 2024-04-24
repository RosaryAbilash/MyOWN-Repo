import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

"""
IMPLEMENT A LOGIC TO SEND A EMAIL TO ADMIN AS MAINTENANCE REQUIRED DATE..
"""

# Sample dataset for load transported
def get_load_data(weight):
    # Generate sample load data (hypothetical)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    load = np.random.randint(1000, 2000, size=len(dates)) * weight
    load_df = pd.DataFrame({'Date': dates, 'Load Transported': load})
    return load_df

# Sample dataset for demand forecasting
def get_demand_data():
    # Generate sample demand data (hypothetical)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    demand = np.random.randint(500, 1500, size=len(dates))
    demand_df = pd.DataFrame({'Date': dates, 'Demand': demand})
    return demand_df

# Sample dataset for energy consumption forecasting
def get_energy_data(weight):
    # Generate sample energy consumption data (hypothetical)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    energy_consumption = np.random.randint(800, 1800, size=len(dates)) * weight
    energy_df = pd.DataFrame({'Date': dates, 'Energy Consumed': energy_consumption})
    return energy_df

# Sample dataset for dynamic maintenance
def get_maintenance_data(weight):
    # Generate sample maintenance data (hypothetical)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='M')
    maintenance_type = ['Regular Inspection', 'Component Replacement', 'Repair']
    maintenance_df = pd.DataFrame({'Date': dates, 'Maintenance Type': np.random.choice(maintenance_type, size=len(dates))})
    maintenance_df['Weight'] = weight
    return maintenance_df

# Function to plot load vs demand graph
def load_vs_demand(load_df, demand_df, weight):
    merged_df = pd.merge(load_df, demand_df, on='Date', how='inner')
    merged_df.set_index('Date', inplace=True)
    fig, ax = plt.subplots()
    merged_df.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")
    st.pyplot(fig)
    st.write("Demand Prediction based on weight:", weight * 75.23)

# Function to plot load vs energy consumption graph
def load_vs_energy_consumption(load_df, energy_df, weight):
    merged_df = pd.merge(load_df, energy_df, on='Date', how='inner')
    merged_df.set_index('Date', inplace=True)
    fig, ax = plt.subplots()
    merged_df.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")
    st.pyplot(fig)
    st.write("Energy Consumption Prediction based on weight:", weight * 250.450)

# Function to plot load vs maintenance needed graph
def load_vs_maintenance(load_df, maintenance_df, weight):
    maintenance_counts = maintenance_df.groupby(maintenance_df['Date'].dt.to_period('M')).size().reset_index(name='Maintenance Needed')
    merged_df = pd.merge(load_df, maintenance_counts, left_on=load_df['Date'].dt.to_period('M'), right_on='Date', how='left').fillna(0)
    merged_df.set_index('Date', inplace=True)
    fig, ax = plt.subplots()
    merged_df.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")
    st.pyplot(fig)
    current_date = datetime.datetime.now()
    future_date = current_date + datetime.timedelta(days=random.randint(1, 365))
    
    st.write("Maintenance Needed based on weight:", future_date)

# # Function to send email to admin with all data
# def send_email(load_df, demand_df, energy_df, maintenance_df):
#     # Prepare email content
#     message = MIMEMultipart()
#     message['From'] = "your_email@gmail.com"
#     message['To'] = "admin_email@gmail.com"
#     message['Subject'] = "Railway Transportation Data"

#     body = f"""\
#     Load Data:
#     {load_df}

#     Demand Data:
#     {demand_df}

#     Energy Consumption Data:
#     {energy_df}

#     Maintenance Data:
#     {maintenance_df}
#     """
#     message.attach(MIMEText(body, 'plain'))

#     # Connect to SMTP server and send email
#     with smtplib.SMTP('smtp.gmail.com', 587) as server:
#         server.starttls()
#         server.login("your_email@gmail.com", "your_password")
#         server.sendmail("your_email@gmail.com", "admin_email@gmail.com", message.as_string())


def send_email(load_df, demand_df, energy_df, maintenance_df, weight):
    current_date = datetime.datetime.now()
    future_date = current_date + datetime.timedelta(days=random.randint(1, 365))
    # Prepare email content
    message = MIMEMultipart()
    message['From'] = 'kioskdocai@gmail.com'
    message['To'] = "ragulm21475@gmail.com"
    message['Subject'] = "Railway Transportation Data"

    body = f"""\

    Maintenance Needed based on weight: {future_date}
    Energy Consumption Prediction based on weight: {weight * 250.450}
    Demand Prediction based on weight: {weight * 75.23}


    Load Data:
    {load_df}

    Demand Data:
    {demand_df}

    Energy Consumption Data:
    {energy_df}

    Maintenance Data:
    {maintenance_df}
    """
    message.attach(MIMEText(body, 'plain'))

    # Connect to SMTP server and send email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        # Replace 'your_email@gmail.com' and 'your_password' with your actual Gmail credentials
        server.login('kioskdocai@gmail.com', 'ttxhzxgkqvblybux')
        server.sendmail('kioskdocai@gmail.com', "ragulm21475@gmail.com", message.as_string())
        st.write("Mail Sent Successfully..")


# Main function to run the Streamlit app
def main():
    # Set page title
    st.title('Railway Wagon Prediction Dashboard')

    # Generate random weight
    weight = random.uniform(0, 3)

    # Get data based on generated weight
    load_df = get_load_data(weight)
    demand_df = get_demand_data()
    energy_df = get_energy_data(weight)
    maintenance_df = get_maintenance_data(weight)

    # Display buttons for different functionalities
    if st.button("Show Weight"):
        st.write(f"Weight: {weight}")

    if st.button("Get Load DATA"):
        st.write(load_df)

    if st.button('Demand Forecasting'):
        load_vs_demand(load_df, demand_df, weight)

    if st.button('Energy Consumption Forecasting'):
        load_vs_energy_consumption(load_df, energy_df, weight)

    if st.button('Predict Maintenance'):
        load_vs_maintenance(load_df, maintenance_df, weight)

    if st.button("Send Email to Admin"):
        send_email(load_df, demand_df, energy_df, maintenance_df,weight)

# Run the Streamlit app
if __name__ == '__main__':
    main()
