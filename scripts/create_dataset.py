#!/usr/bin/python3
import os, glob
import datetime

import xmltodict as xd

import numpy as np
import pandas as pd
import h5py

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing

# lets make a little data set for fun...
mh_dir = os.path.abspath('./db/mh_data/')
mh_cases = glob.glob(os.path.join(mh_dir, '*'))

# sample = os.path.abspath('./db/asac_data/case10.xml') >> TODO: we will need
# to make modifications for this dataset

def xml_parser(xml_path):

    with open(xml_path) as fd:
            doc = xd.parse(fd.read())
            fd.close()

    raw_db = doc['anaesthetic']['data']['var']
    print("FILE READ")


    for i in raw_db[3:]:
        name = i['vaname']

        times = str(i['vatimes']).replace('None','000000').split(',')
        values = str(i['vavalues']).replace('NA','nan').split(',')

        times = np.asarray(times)
        values = np.asarray(values).astype('float')

        var_df = pd.DataFrame(data = {'time' : times, name : values})

        if 'full_df' in locals():
            full_df = full_df.join(var_df.set_index('time'), on='time')
        else:
            full_df = var_df

    print("XML PARSED")

    return full_df

class Input: # an input struct
    pass


db = [] # list of all input structs

def delta_spo2(spo2_arr):
    # compute the difference between the maximum value 
    max_val = max(spo2_arr)
    min_val = min(spo2_arr)
    d = max_val - min_val

    return d

# a sample will be 6 entries (=60 seconds) of every datapoint to determine if
# there will be a change in spo2 in the next 60 seconds
# spo2, hr, 
def data_generator(patient_df):
    # slice the df into array of 6 element dfs
    interval_df = []

    for i in range(patient_df.shape[0]):
        if (i+1) % 6 == 0:
            # split every 6 timestamp (60 seconds)
            a = i - 5
            interval_df.append(patient_df[a:i+1])
        else:
            continue

    # compute spo2 delta
    for i in range(len(interval_df)):
        sample = Input()
        sample.x = np.asarray(interval_df[i].unstack()) # vector of input data from 
        try:
            sample.d = delta_spo2(interval_df[i+1]['spo2.SpO2'])
        except:
            print("end of dataset")
            break
        # label
        if sample.d > 0.011:
            sample.y = 1
        else:
            sample.y = 0 
        db.append(sample)

    return db


# parse every xml file and save each to a separate h5 file for future use
# spo2.SpO2, co2.et, ecg.hr, nibp.sys, nibp.dia
def mk_npy():
    for i in mh_cases:
        print(i)
        df = xml_parser(i)

        # for all features simply use df
        # spo2.SpO2, co2.et, ecg.hr, nibp.sys, nibp.dia
        df2 = pd.DataFrame(df,
                columns=['ecg.hr',
                    'co2.et', 'nibp.sys',
                    'nibp.dia', 'spo2.SpO2']
                )

        df2 = df2[np.abs(df2-df2.mean()) <= (3*df2.std())]
        df2 = df2.dropna()

        # scale the values between 1-0 the data by patient....
        x = df2.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df2 = pd.DataFrame(x_scaled, columns=df2.columns)

        data_generator(df2)

    X = []
    Y = []

    for i in db:
        X.append(i.x)
        Y.append(i.y)

    X = np.asarray(X).astype('float')
    Y = np.asarray(Y).astype('int')
    print("stable: " + str(np.sum(Y == 0)))
    print("unstable: " + str(np.sum(Y == 1)))

    np.save("x3.npy", X)
    np.save("y3.npy", Y)



mk_npy()

# boom load it...
#X = np.load("x.npy", X) # (3740, 306)
#Y = np.load("y.npy", Y) # (3740,)

