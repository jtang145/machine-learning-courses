import pandas as pd
import numpy as np
import datetime

def load_data():
    # Sales Data
    all_data_file = "./data/train.csv"
    data = pd.read_csv(all_data_file, dtype={"StateHoliday":np.str},
                      parse_dates=['Date'])
    # Format Sales data
    data.drop(['DayOfWeek'],axis = 1, inplace = True)
    data.fillna('0')
    return data

# View 'size' of store's
def viewSalesData(data, keys=[], size = 10):
    stores = np.random.choice(len(data['Store'].unique()),size)
    view_frames = frame.drop(['Customers','Open','Promo','StateHoliday','SchoolHoliday'], axis =1)
    for i in stores:
