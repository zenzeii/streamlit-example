import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


# Streamlit app title
st.title('Sales Forecasting')

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=['xlsx'])
if uploaded_file is not None:
    selected_sheet = st.selectbox('Select sheet', pd.ExcelFile(uploaded_file).sheet_names, index=1)
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    # Display the dataframe
    st.write("Data Preview:")
    st.write(df.head())

    # Check if default columns are present in the dataframe
    if "sales_date" in df.columns and "gross_sales" in df.columns:
        selected_columns = st.multiselect('Select 2 columns (ds and y)', df.columns, default=["sales_date", "gross_sales"])
    else:
        selected_columns = st.multiselect('Select 2 columns (ds and y)', df.columns)

    if len(selected_columns) == 2:
        df = df[selected_columns]
        df = df.groupby(selected_columns[0], as_index=False)[selected_columns[1]].sum()
        df = pd.DataFrame({'ds': df[selected_columns[0]], 'y': df[selected_columns[1]]})
        df.columns = ['ds', 'y']  # Rename for Prophet compatibility

        # Split data
        df_train = df[df['ds'] < '2021-01-01']
        df_test = df[(df['ds'] >= '2021-01-01') & (df['ds'] < '2022-01-01')]

        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(df_train)

        # Predict
        future_dates = model.make_future_dataframe(periods=24, freq='M')
        forecast = model.predict(future_dates)

        # Error metrics
        forecasted_2021 = forecast[(forecast['ds'] >= '2021-01-01') & (forecast['ds'] < '2022-01-01')]
        forecasted_2021_2022 = forecast[(forecast['ds'] >= '2021-01-01') & (forecast['ds'] < '2023-01-01')]
        forecasted_2022 = forecast[(forecast['ds'] >= '2021-12-01') & (forecast['ds'] < '2023-01-01')]

        mae = mean_absolute_error(df_test['y'], forecasted_2021['yhat'])
        rmse = np.sqrt(mean_squared_error(df_test['y'], forecasted_2021['yhat']))

        # Display metrics with explanations
        st.write(f"MAE for 2021: {mae:.2f}")
        st.write("Mean Absolute Error (MAE) is the average of the absolute errors between the predicted and actual values. It measures the average magnitude of errors in a set of predictions, without considering their direction. A lower MAE value indicates a better fit of the model to the data.")

        st.write(f"RMSE for 2021: {rmse:.2f}")
        st.write("Root Mean Squared Error (RMSE) is the square root of the average of squared differences between prediction and actual observation. It measures how concentrated the data is around the line of best fit. Unlike MAE, RMSE gives a relatively high weight to large errors, meaning it can highlight large errors better than MAE. A lower RMSE value indicates a better fit of the model to the data.")


        # Plotting with Plotly
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Actual Sales - using blue for the actual sales line
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual Sales', mode='lines', line=dict(color='blue')), secondary_y=False)

        # Predicted Sales - using red for the predicted sales line
        fig.add_trace(go.Scatter(x=forecasted_2021_2022['ds'], y=forecasted_2021_2022['yhat'], name='Predicted Sales',mode='lines', line=dict(color='orange')), secondary_y=False)

        # Fill between for forecast interval in 2022 - using gray for the confidence interval fill
        fig.add_trace(go.Scatter(x=forecasted_2022['ds'], y=forecasted_2022['yhat_upper'], mode='lines', fill=None, name='Upper Confidence Interval', line=dict(width=0)), secondary_y=False)
        fig.add_trace(go.Scatter(x=forecasted_2022['ds'], y=forecasted_2022['yhat_lower'], mode='lines', fill='tonexty', name='Lower Confidence Interval', line=dict(width=0)), secondary_y=False)

        # Forecast Start Line - using orange for the forecast start line, matching your matplotlib style
        fig.add_vline(x=pd.to_datetime('2022-01-01').timestamp() * 1000, line=dict(color="orange", dash='dash'),name='Forecast Start')

        # Layout adjustments to match matplotlib style
        fig.update_layout(title='Sales Forecast', xaxis_title='Date', yaxis_title='Sales', legend=dict(y=1, x=1, bgcolor='rgba(255,255,255,0.5)'))
        fig.update_yaxes(title_text="Sales", secondary_y=False)

        st.plotly_chart(fig)

        # Display a loading message while performing cross-validation
        with st.spinner('Performing cross-validation... Please wait'):
            df_cv = cross_validation(model, initial='365 days', period='90 days', horizon='180 days')
            df_p = performance_metrics(df_cv)

        # Once completed, show the performance metrics and the plot
        st.write("Cross-validation performance metrics:")
        st.write(df_p.head())

        with st.spinner('Generating plot...'):
            fig = plot_cross_validation_metric(df_cv, metric='mae')
            st.plotly_chart(fig)

    else:
        st.error('Please select exactly two columns.')

