from telnetlib import X3PAD
from urllib.request import AbstractBasicAuthHandler
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import time
from time import sleep
import snscrape.modules.twitter as sntwitter
import yfinance as yf
from yahoofinancials import YahooFinancials
from datetime import date

#viz
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from PIL import Image

#modeling
from sklearn.preprocessing import MinMaxScaler

### Create the Stacked LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error

#def main(): #layout
#st.set_page_config(layout="wide")
#menu = ['Playbook', 'About']
#choice = st.sidebar.selectbox('Menu',menu)


def main():
    #def main(): #layout
    st.set_page_config(layout="wide")
    st.markdown("## Michel´s Playbook - Stock Forecast App")


    menu = ['Select Stock','Overview', 'Model','About']    
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'TSLA')


    selected_stock = st.selectbox('Select your stock of analysis', stocks)


    fin_data = yf.Ticker(selected_stock)

    fin_data = yf.Ticker(selected_stock)

    name = fin_data.info['shortName']

    image5 = Image.open('pic5.png')
    st.sidebar.image(image5)

    choice = st.sidebar.selectbox('Menu',menu)
    logo = fin_data.info['logo_url']
    website = fin_data.info['website']
    sector = fin_data.info['sector']
    country = fin_data.info['country']
    summary = fin_data.info['longBusinessSummary']
    
    st.markdown('After you have selected your stock, please navigate the menu')
    
    if choice == 'Overview':

        
            ## Main Title
        st.markdown(f'''![Foo]({logo}) ({website})   

                        {sector} located in {country}''')
        
        
        st.subheader('Company Description')
        st.write(summary)

        

        st.subheader(f'Last-10 Days Performance for {name}')
        col1,col2 = st.columns([2,1]) #first column to be 2 times bigger than 1st column

        col1.subheader('Including Financial Indicators')

    
        #n_days = ('15d','30d','60d')
        #selected_days = st.selectbox('Select stock to analyze', n_days)

        #@st.cache

        data_load_state = st.text('Loading data...')
        
        def load_data(ticker):
            data_yf = yf.download(  # or pdr.get_data_yahoo(...
                    # tickers list or string as well
                    tickers = selected_stock,

                    # use "period" instead of start/end
                    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                    # (optional, default is '1mo')
                    period = "5y",

                    # fetch data by interval (including intraday if period < 60 days)
                    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                    # (optional, default is '1d')
                    interval = "1d",

                    # group by ticker (to access via data['SPY'])
                    # (optional, default is 'column')
                    group_by = 'ticker',

                    # adjust all OHLC automatically, Auto Adjust will overwrite the Close price by Adj Close. 
                    # (optional, default is False)
                    auto_adjust = True,

                    # download pre/post regular market hours data
                    # (optional, default is False)
                    prepost = True,

                    # use threads for mass downloading? (True/False/Integer)
                    # (optional, default is True)
                    threads = True,

                    # proxy URL scheme use use when downloading?
                    # (optional, default is None)
                    proxy = None
                )
            return data_yf

        

        data = load_data(selected_stock)

        data['MarketCap'] = data['Open'] * data['Volume']

        data['forwardEPS'] = fin_data.info['forwardEps']
        data['trailingEPS'] = fin_data.info['trailingEps']

        data['ForwardPE'] = data['Open'] / fin_data.info['forwardEps']
        data['TrailingPE'] = data['Open'] / fin_data.info['trailingEps']

        # PEG RATIOS 

        data['ForwardPEG'] = data['ForwardPE'] / fin_data.info['earningsGrowth']
        data['TrailingPEG'] = data['TrailingPE'] / fin_data.info['earningsGrowth']


        data.drop(['forwardEPS','trailingEPS'], axis = 1, inplace = True)

        data.reset_index(inplace = True)

        data.to_csv('data_clean.csv', index = False)

        data_load_state.text('Data is now loaded!')

                #Plot raw data
        def plot_raw_data():
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text=f'{selected_stock} 5Y Close Price', xaxis_rangeslider_visible=True)
            
            col2.plotly_chart(fig)

        col1.write(data.tail(15))

        plot_raw_data()


        #SECOND ONE
    elif choice == 'Model':

        st.subheader(f'Let´s have a look at {name} stock with an additional days forecasted')
        data_load_state2 = st.text('Loading model...')
        #MODELING

        df = pd.read_csv('data_clean.csv',na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)

        df1 =df[['Close']]

        #scaling

        scaler=MinMaxScaler(feature_range=(0,1))
        df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

        #split training and testt

        training_size=int(len(df1)*0.75)
        test_size=len(df1)-training_size
        train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


        # convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])

            #return np.array(dataX), np.array(dataY)
        

        
        # reshape into X=t,t+1,t+2,t+3 and Y=t+4

        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)


        # reshape input to be [samples, time steps, features] which is required for LSTM

        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        #create 

        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')

        #fitting model

        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

        ### Lets Do the prediction and check performance metrics

        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        ##Transformback to original form

        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)

        ### Declaring plotting variables. 

        # shift train predictions for plotting
        look_back=100
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
        # plot baseline and predictions

        x_input=test_data[215:].reshape(1,-1)
        x_input.shape


        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        lst_output=[]
        n_steps=100
        i=0

        data_load_state2 = st.text('Model Trained.')

        opt2 = (0 , 15, 30, 45, 60, 75, 90)
        days = st.selectbox('Select the days you would like to forecast', opt2)

        time.sleep(10)
      
        if days == 0:
            print('Please select your days')
            time.sleep(30) #chance to select

        else:
            time.sleep(10)#in case they change their mind
            st.text(f'{days} days selected')
            data_load_state3 = st.text('Now the prediction model is loading... Please wait!')
            while(i<days):
                
                if(len(temp_input)>100):
                    
                    x_input=np.array(temp_input[1:])
                    
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    
                    yhat = model.predict(x_input, verbose=0)
                    
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    
                    temp_input.extend(yhat[0].tolist())
                    
                    lst_output.extend(yhat.tolist())
                    i=i+1

            day_new =np.arange(1,101)
            day_pred =np.arange(101,131)

            df3=df1.tolist()
            df3.extend(lst_output)

            df3=scaler.inverse_transform(df3).tolist()

            data_load_state3 = st.text('Completed, let´s have a look at the predictions')

            st.line_chart(data = df3, height = 200, width = 20, use_container_width= True)

            df_forecast = pd.DataFrame(df3)
            pred = df_forecast.iloc[1259:1289]

            pred2 = pred.rename(columns = {0:'Close Price'})
            pred2.reset_index(inplace = True)

            pred2 = pred2[['Close Price']]

            st.line_chart(data = pred2)


            
            fc_growth = round((((pred2.iloc[29][0] / pred2.iloc[0][0]) - 1) * 100), 2)

            if fc_growth > 0:
                st.subheader(f'According to the model, in {days} the growth is {fc_growth}%! You should Buy!')
            else:
                st.subheader(f'According to the model, in {days} the growth is {fc_growth}%! You should NOT Buy!')
            
    
    elif choice == 'About':
        st.markdown("## Michel´s Playbook - Stock Forecast App")
        image4 = Image.open('pic4.png')
        st.image(image4, caption='Information')
    else:
        st.markdown('---------------')



if __name__ == '__main__':
    main()

#Absaaaa

