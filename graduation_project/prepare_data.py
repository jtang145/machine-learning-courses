import pickle
import csv
from datetime import datetime
import time as tm
import numpy as np
from sklearn import preprocessing
import random
random.seed(4)


def csv2dicts(csvfile):
    data = []
    keys = []
    for row_index, row in enumerate(csvfile):
        # skip title
        if row_index == 0:
            keys = row
            print("Titles:")
            print(row)
            continue
        # if row_index % 10000 == 0:
        #     print(row_index)
        data.append({key: value for key, value in zip(keys, row)})
    return data

def csv2dictsWithSplitDate(csvfile):
    data = []
    keys = []
    for row_index, row in enumerate(csvfile):
        # skip title
        if row_index == 0:
            keys = row
            print("Titles:")
            print(row)
            continue
        #if row_index==10:
            #print(row[2])
        item = {key: value for key, value in zip(keys, row)}
        data.append(item)
        item["tm_date"] = tm.strptime(row[2],"%Y-%m-%d")
        item["tm_time"] = item["tm_date"].time()
    return data

def set_nan_as_string(data, replace_str='0'):
    for i, x in enumerate(data):
        for key, value in x.items():
            if value == '':
                x[key] = replace_str
        data[i] = x

def flatDate(data):
    for index,val in enumerate(data):
        for key, value in val.items():
            if key == 'Date':
                date_of_item = value.split('-')
                val['Date_Year'] = date_of_item[0]
                val['Date_Month'] = date_of_item[1]
                val['Date_Day'] = date_of_item[2]
        data[index] = val

def read_data_as_dicts():
    train_data = "./data/train.csv"
    store_data = "./data/store.csv"
    store_states = './data/store_states.csv'
    with open(train_data) as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        with open('train_data.pickle', "wb") as f:
            data = csv2dicts(data)
            # Let order by date descending
            data = data[::-1]
            pickle.dump(data, f, -1)
            print("3 samples:")
            print(data[:3])
            print("")


    with open(store_data) as csvfile, open(store_states) as csvfile2:
        data = csv.reader(csvfile, delimiter=',')
        state_data = csv.reader(csvfile2, delimiter=',')
        with open('store_data.pickle', 'wb') as f:
            data = csv2dicts(data)
            state_data = csv2dicts(state_data)
            set_nan_as_string(data)
            for index, val in enumerate(data):
                state = state_data[index]
                val['State'] = state['State']
                data[index] = val
            pickle.dump(data, f, -1)
            print("2 samples:")
            print(data[:2])
            print("")
    print("Extract data from csv file completed.")
    print("")

def toString():
    return "This is prepare_data file."



def feature_list(store_data, record):
    dt = datetime.strptime(record['Date'], '%Y-%m-%d')
    store_index = int(record['Store'])
    year = dt.year
    month = dt.month
    day = dt.day
    day_of_week = int(record['DayOfWeek'])
    try:
        store_open = int(record['Open'])
    except:
        store_open = 1

    promo = int(record['Promo'])

    return [store_open,
            store_index,
            day_of_week,
            promo,
            year,
            month,
            day,
            store_data[store_index - 1]['State']
            ]

def prepare_data_for_process():
    print("Prepare data from processning")
    with open('train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)
        num_records = len(train_data)
    with open('store_data.pickle', 'rb') as f:
        store_data = pickle.load(f)

    train_data_X = []
    train_data_y = []

    for record in train_data:
        if record['Sales'] != '0' and record['Open'] != '':
            fl = feature_list(store_data, record)
            train_data_X.append(fl)
            train_data_y.append(int(record['Sales']))
    print("Number of train datapoints: ", len(train_data_y))

    print(min(train_data_y), max(train_data_y))

    full_X = train_data_X
    full_X = np.array(full_X)
    train_data_X = np.array(train_data_X)
    les = []
    for i in range(train_data_X.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(full_X[:, i])
        les.append(le)
        train_data_X[:, i] = le.transform(train_data_X[:, i])

    with open('les.pickle', 'wb') as f:
        pickle.dump(les, f, -1)

    train_data_X = train_data_X.astype(int)
    train_data_y = np.array(train_data_y)

    with open('feature_train_data.pickle', 'wb') as f:
        pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])
    return train_data_X, train_data_y


read_data_as_dicts()
# prepare_data_for_process()
