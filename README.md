# MichelsPlaybook - Stock Prediction App 
#### Final Project for Ironhack PTJan2022

![pic4](https://user-images.githubusercontent.com/76560772/176016781-f9ae110b-af67-4616-be7d-ade355b3bfc8.png)

## Data Gathering
![image](https://user-images.githubusercontent.com/76560772/176017372-8ceac35b-2ef1-43f5-9fca-69683d51d7da.png)

Using the library from Yahoo Finance, yfinance, I was able to obtain live stock market data alongside other valuable informatiohn:
- High
- Low
- Open
- Close
- EPS
- Revenue
- Financial Ratios

This data was then cleaned to obtain a specific stock ticker. 

## Modeling

- Use of LSTM as selected model. ARIMA and SARIMAX tested.
- Selection of train and test data according to the selected amount of data. 
- Training of the model. 
- Use model to predict future values. 
- Future value prediction. 

## Visualization - Streamlit

- **Stock Selection**: Here the user will select one stock from multiple options. 
- **Overview**: Information of the company with additional stock and ratio data provided, alongside chart.
- **Model**: The user will start the model and select days of future prediction. Also, the predicted growth is given and a reccomendation alongside it. 
- **About**: Information (slide above). 



