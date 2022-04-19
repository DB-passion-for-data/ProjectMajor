
import streamlit as st
import time
st.title("Predicting Failure of Aircraft Engine")

st.image("https://s30121.pcdn.co/wp-content/uploads/2019/03/p1d48jfreovehtnhvj51ld7ndf6.jpg")
st.sidebar.subheader("Data Description")
st.sidebar.write("This data set was generated with the C-MAPSS simulator. C-MAPSS stands for 'Commercial Modular Aero-Propulsion System Simulation' and it is a tool for the simulation of realistic large commercial turbofan engine data. Each flight is a combination of a series of flight conditions with a reasonable linear transition period to allow the engine to change from one flight condition to the next. The flight conditions are arranged to cover a typical ascent from sea level to 35K ft and descent back down to sea level. The fault was injected at a given time in one of the flights and persists throughout the remaining flights, effectively increasing the age of the engine. The intent is to identify which flight and when in the flight the fault occurred.")

st.sidebar.subheader("How Data Was Acquired")
st.sidebar.write("The data provided is from a high fidelity system level engine simulation designed to simulate nominal and fault engine degradation over a series of flights. The simulated data was created with a Matlab Simulink tool called C-MAPSS.")

st.sidebar.subheader("Sample Rates and Parameter Description")
st.sidebar.write("The flights are full flight recordings sampled at 1 Hz and consist of 30 engine and flight condition parameters. Each flight contains 7 unique flight conditions for an approximately 90 min flight including ascent to cruise at 35K ft and descent back to sea level. The parameters for each flight are the flight conditions, health indicators, measurement temperatures and pressure measurements.")

st.sidebar.write("Download the Paper related to Data Generation here")
with open("files/Damage Propagation Modeling.pdf", "rb") as file:
     btn = st.sidebar.download_button(
             label="Damage Propagation Modeling",
             data=file,
             file_name="Damage Propagation Modeling.pdf",
             mime="text/pdf"
           )
#st.sidebar.download_button(label = "Damage Propagation Modeling",data = files\Damage Propagation Modeling.pdf)
st.write("""Predictive Maintenance techniques are used to determine the condition of
         an equipment to plan the maintenance/failure ahead of its time. This is
         very useful as the equipment downtime cost can be reduced significantly.
         The objective of this project is to predict the failure of machine in
         upcoming n days.""")

st.subheader("Structure of turbofan engine in C-MAPSS simulation")
st.image("https://www.researchgate.net/profile/Cheng-Yiwei-2/publication/336606215/figure/fig3/AS:814971399200771@1571315731247/Structure-of-turbofan-engine-in-C-MAPSS-simulation.png")

st.subheader("Example of data simulated from CMAPSS")
st.image("https://www.researchgate.net/publication/329772218/figure/fig2/AS:705397212078080@1545191209695/An-example-from-C-MAPSS-a-turbofan-engine-degradation-dataset-from-NASA-26-showing.png")

st.write("This is the type of data generated from C-MAPSS simulation. Our aim is to find conditions of failure based upon the values of sensor and the cycle ongoing.")
unit_number = st.number_input("Enter unit_number",value=1)
cycle = st.number_input("Enter the ongoing cycle",value = 1)
op1 = st.number_input("Enter the operational setting 1",value =-0.080460)
op2 = st.number_input("Enter the operational setting 2",value =-0.666667)
sm1= st.number_input("Sensor Measure 1",value =-0.632530)
sm2= st.number_input("Sensor Measure 2",value =-0.186396)
sm3= st.number_input("Sensor Measure 3",value = -0.380486)
sm4= st.number_input("Sensor Measure 4",value =0.452496)
sm5= st.number_input("Sensor Measure 5",value =-0.515152)
sm6= st.number_input("Sensor Measure 6",value = -0.780490)
sm7= st.number_input("Sensor Measure 7",value = -0.261905)
sm8= st.number_input("Sensor Measure 8",value = 0.266525)
sm9= st.number_input("Sensor Measure 9",value =-0.588235)
sm10= st.number_input("Sensor Measure 10",value = -0.272028)
sm11= st.number_input("Sensor Measure 11",value =-0.333333)
sm12= st.number_input("Sensor Measure 12",value =0.426357)
sm13= st.number_input("Sensor Measure 13",value = 0.449323) 
#rul = st.number_input("Enter the Remaining Useful Life")

#making the code
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler #to normalize data
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

#for deep learning
import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Masking, Dropout
from keras.callbacks import History
from keras import callbacks

def prepare_data(drop_cols = True):
    dependent_var = ['RUL']
    index_columns_names =  ["UnitNumber","Cycle"]
    operational_settings_columns_names = ["OpSet"+str(i) for i in range(1,4)]
    sensor_measure_columns_names =["SensorMeasure"+str(i) for i in range(1,22)]
    input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names

    cols_to_drop = ['OpSet3', 'SensorMeasure1', 'SensorMeasure5', 'SensorMeasure6', 'SensorMeasure10', 'SensorMeasure14',
     'SensorMeasure16', 'SensorMeasure18', 'SensorMeasure19']

    df_train = pd.read_csv('data/train_FD001.txt',delim_whitespace=True,names=input_file_column_names)

    rul = pd.DataFrame(df_train.groupby('UnitNumber')['Cycle'].max()).reset_index()
    rul.columns = ['UnitNumber', 'max']
    df_train = df_train.merge(rul, on=['UnitNumber'], how='left')
    df_train['RUL'] = df_train['max'] - df_train['Cycle']
    df_train.drop('max', axis=1, inplace=True)

    df_test = pd.read_csv('data/test_FD001.txt', delim_whitespace=True, names=input_file_column_names)
    
    if(drop_cols == True):
        df_train = df_train.drop(cols_to_drop, axis = 1)
        df_test = df_test.drop(cols_to_drop, axis = 1)

    y_true = pd.read_csv('data/RUL_FD001.txt', delim_whitespace=True,names=["RUL"])
    y_true["UnitNumber"] = y_true.index
    
    return df_train, df_test, y_true

df_train, df_test, y_true = prepare_data()
feats = df_train.columns.drop(['UnitNumber', 'Cycle', 'RUL'])

min_max_scaler = MinMaxScaler(feature_range=(-1,1))

df_train[feats] = min_max_scaler.fit_transform(df_train[feats])
df_test[feats] = min_max_scaler.transform(df_test[feats])

def gen_train(id_df, seq_length, seq_cols):
    """
        function to prepare train data into (samples, time steps, features)
        id_df = train dataframe
        seq_length = look back period
        seq_cols = feature columns
    """
        
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)

def gen_target(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length-1:num_elements+1]

def gen_test(id_df, seq_length, seq_cols, mask_value):
    """"
        function to prepare test data into (samples, time steps, features)
        function only returns last sequence of data for every unit
        id_df = test dataframe
        seq_length = look back period
        seq_cols = feature columns
    """
    df_mask = pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    df_mask[:] = mask_value
    
    id_df = df_mask.append(id_df,ignore_index=True)
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    
    start = num_elements-seq_length
    stop = num_elements
    
    lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)

sequence_length = 50 #predicting using last 30 cycle values
mask_value = 0

#x_train=np.concatenate(list(list(gen_train(df_train[df_train['UnitNumber']==unit], sequence_length, feats)) for unit in df_train['UnitNumber'].unique()))

x_input = np.array([unit_number,cycle,op1,op2,sm1,sm2,sm3,sm4,sm5,sm6,sm7,sm8,sm9,sm10,sm11,sm12,sm13]).reshape(1,17)
df = pd.DataFrame(data = x_input,columns=['UnitNumber', 'Cycle', 'OpSet1', 'OpSet2', 'SensorMeasure2',
       'SensorMeasure3', 'SensorMeasure4', 'SensorMeasure7', 'SensorMeasure8',
       'SensorMeasure9', 'SensorMeasure11', 'SensorMeasure12',
       'SensorMeasure13', 'SensorMeasure15', 'SensorMeasure17',
       'SensorMeasure20', 'SensorMeasure21'])
df_test1 = df_test.append(df)
x_input=np.concatenate(list(list(gen_test(df_test1[df_test1['UnitNumber']==unit], sequence_length, feats,mask_value)) for unit in df_test1['UnitNumber'].unique()))
from keras.models import load_model
predictor_model = load_model('my_model')
if st.button('Predict now'):
     with st.spinner(text = 'Predicting engine remaining useful life....'):
          time.sleep(4)
          predictor_model.compile()
  #prediction = (predictor_model.predict((np.array(x_input).reshape(1,50,15))) > 0.5).astype("int32")
  
  #a = (predictor_model.predict(x_input)>0.5)
          prediction = predictor_model(x_input)
          a = prediction[-1].numpy()
          days = str(round(a[0]))
          no_days = days + " days"
          st.write("Engine's remaining useful life is about ",st.subheader(no_days))
          
