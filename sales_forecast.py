# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime
import warnings
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import kagglehub
import plotly.offline as py

warnings.filterwarnings('ignore')

# Step 1: Load the Dataset
path = kagglehub.dataset_download("kyanyoga/sample-sales-data")
df = pd.read_csv(f"{path}/sales_data_sample.csv", encoding='latin1')

# Step 2: Initial Inspection
print("\nDataFrame Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Check for null values
print("\nMissing values:")
print(df.isnull().sum())

# Step 3: Preprocessing
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df = df.sort_values('ORDERDATE')

# Step 4: Aggregate daily sales
daily_sales = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
daily_sales.columns = ['ds', 'y']

# Visualize original sales time series
plt.figure(figsize=(15, 5))
plt.plot(daily_sales['ds'], daily_sales['y'])
plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("original_timeseries.png")
plt.close()

# Plot rolling average (optional)
daily_sales['rolling_mean_7'] = daily_sales['y'].rolling(window=7).mean()
daily_sales['rolling_mean_30'] = daily_sales['y'].rolling(window=30).mean()

plt.figure(figsize=(15, 5))
plt.plot(daily_sales['ds'], daily_sales['y'], label='Actual Sales', alpha=0.4)
plt.plot(daily_sales['ds'], daily_sales['rolling_mean_7'], label='7-Day MA')
plt.plot(daily_sales['ds'], daily_sales['rolling_mean_30'], label='30-Day MA')
plt.title("Sales with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("moving_averages.png")
plt.close()

# Step 5: Train Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, changepoint_prior_scale=0.05)

model.fit(daily_sales[['ds', 'y']])

# Step 6: Forecasting
future_dates = model.make_future_dataframe(periods=90)
forecast = model.predict(future_dates)

# Step 7: Plot Forecast
fig1 = model.plot(forecast)
plt.title("Forecast vs Actual Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
fig1.savefig("sales_forecast.png")
plt.close()

# Step 8: Forecast Components
fig2 = model.plot_components(forecast)
plt.tight_layout()
fig2.savefig("forecast_components.png")
plt.close()


# Step 9: Forecast Evaluation
historical_forecast = forecast[forecast['ds'].isin(daily_sales['ds'])]
actual = daily_sales['y']
predicted = historical_forecast['yhat']

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("\nForecast Accuracy Metrics:")
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Step 10: Confusion Matrix based on categories
def create_sales_categories(sales_data, num_categories=5):
    percentiles = np.linspace(0, 100, num_categories + 1)
    thresholds = np.percentile(sales_data, percentiles)
    return np.digitize(sales_data, thresholds[1:-1])

actual_categories = create_sales_categories(actual)
predicted_categories = create_sales_categories(predicted)

cm = confusion_matrix(actual_categories, predicted_categories)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Cat {i+1}' for i in range(5)],
            yticklabels=[f'Cat {i+1}' for i in range(5)])
plt.title("Sales Category Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

category_accuracy = cm.diagonal().sum() / cm.sum()
print(f"\nCategory-wise Accuracy: {category_accuracy:.2%}")

# Step 11: Cross-validation (optional)
df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='90 days')
df_p = performance_metrics(df_cv)

print("\nCross-validation Performance Metrics:")
print(df_p[['horizon', 'mae', 'rmse', 'mape']].head())

# Step 12: Save forecast
forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_results.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
forecast_results.to_csv("sales_forecast_results.csv", index=False)

# Confirmation message
print("\nAll outputs saved:")
print("- 'sales_forecast.png' for forecast")
print("- 'forecast_components.png' for seasonality")
print("- 'confusion_matrix.png' for category accuracy")
print("- 'moving_averages.png' for sales trends")
print("- 'sales_forecast_results.csv' for forecast data")
