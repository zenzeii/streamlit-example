import streamlit as st
import pandas as pd
import plotly.express as px

# Set the title of the app
st.title('Time Series Data Visualizer')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a file", type=['csv'])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("Data Preview:")
    st.write(df.head())

    # Ensure there are at least two columns to prevent index errors
    if df.shape[1] >= 2:
        # Let the user select the column for the X-axis (time) and Y-axis (value)
        # Let the user select the column for the X-axis (time) and Y-axis (value)
        x_axis = st.selectbox('Select the X-axis (Time):', options=df.columns, index=0)
        y_axis = st.selectbox('Select the Y-axis (Value):', options=df.columns, index=1)
    else:
        # Fallback if there's only one column or dataframe is incorrectly formatted
        st.error("The uploaded file must have at least two columns.")
        st.stop()


    # Plot the time series data
    fig = px.line(df, x=x_axis, y=y_axis, title=f'Time Series of {y_axis}')
    st.plotly_chart(fig, use_container_width=True)

