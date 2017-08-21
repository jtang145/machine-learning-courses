import pickle
import csv
import datetime
import time as tm


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
        item["tm_date"]= tm.strptime(row[2],"%Y-%m-%d")
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


train_data = "./data/train.csv"
store_data = "./data/store.csv"
store_states = './data/store_states.csv'

with open(train_data) as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    with open('train_data.pickle', 'wb') as f:
        data = csv2dictsWithSplitDate(data)
        # flatDate(data)
        data = data[::-1]
        pickle.dump(data, f, -1)
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
        print(data[:2])
        print("")

def toString():
    return "This is prepare_data file."
