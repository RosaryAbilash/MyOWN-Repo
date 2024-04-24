import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
IMPLEMENT A LOGIC BASED ON THE CURRENT CARGO WEIGHT WHAT IS THE DEMAND NOW AND DISPLAY THE PREDICTED DEMAND..
IMPLEMENT A LOGIC BASED ON THE CURRENT WEIGHT WHAT IS THE  FUEL CONSUMPTION WILL AND DISPLAY THE PREDICTED FUEL CONSUMPTION..
IMPLEMENT A LOGIC TO SEND A EMAIL TO ADMIN AS MAINTENANCE REQUIRED DATE..
        
"""


def get_weight():
    pass

# Sample dataset for load transported
def get_load_data():
    # Generate sample load data (hypothetical)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    load = np.random.randint(1000, 2000, size=len(dates))
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
def get_energy_data():
    # Generate sample energy consumption data (hypothetical)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    energy_consumption = np.random.randint(800, 1800, size=len(dates))
    energy_df = pd.DataFrame({'Date': dates, 'Energy Consumed': energy_consumption})
    return energy_df

# Sample dataset for dynamic maintenance
def get_maintenance_data():
    # Generate sample maintenance data (hypothetical)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='M')
    maintenance_type = ['Regular Inspection', 'Component Replacement', 'Repair']
    maintenance_df = pd.DataFrame({'Date': dates, 'Maintenance Type': np.random.choice(maintenance_type, size=len(dates))})
    return maintenance_df

# Function to plot load vs demand graph
def load_vs_demand():
    load_df = get_load_data()
    demand_df = get_demand_data()
    merged_df = pd.merge(load_df, demand_df, on='Date', how='inner')
    merged_df.set_index('Date', inplace=True)
    st.line_chart(merged_df)

# Function to plot load vs energy consumption graph
def load_vs_energy_consumption():
    load_df = get_load_data()
    energy_df = get_energy_data()
    merged_df = pd.merge(load_df, energy_df, on='Date', how='inner')
    merged_df.set_index('Date', inplace=True)
    st.line_chart(merged_df)

# Function to plot load vs maintenance needed graph
def load_vs_maintenance():
    load_df = get_load_data()
    maintenance_df = get_maintenance_data()
    maintenance_counts = maintenance_df.groupby(maintenance_df['Date'].dt.to_period('M')).size().reset_index(name='Maintenance Needed')
    merged_df = pd.merge(load_df, maintenance_counts, left_on=load_df['Date'].dt.to_period('M'), right_on='Date', how='left').fillna(0)
    merged_df.set_index('Date', inplace=True)
    st.line_chart(merged_df)

# Main function to run the Streamlit app
def main():
    # Set page title
    st.title('Railway Transportation Dashboard')

    if st.button("Get Load DATA"):
        get_weight()

    # Option for load vs demand graph
    if st.button('Demand Forecasting'):
        load_vs_demand()
        


    # Option for load vs energy consumption graph
    if st.button('Energy Consumption Forecasting'):
        load_vs_energy_consumption()
        
    # Option for load vs maintenance needed graph
    if st.button('Predict Maintenance'):
        pass

# Run the Streamlit app
if __name__ == '__main__':
    main()
