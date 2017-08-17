import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math

#%matplotlib inline

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
# group: The supported group catagory, refer to https://chrisalbon.com/python/pandas_group_data_by_time.html
# aggregation: the aggregation method: sum, mean
# size: the Store counts to be viewed, pick up randomly
def viewSalesDataOverTime(data, group, aggregation= 'sum', size = 10):
    stores = np.random.choice(len(data['Store'].unique()),size)
    view_frames = data.drop(['Customers','Open','Promo','SchoolHoliday','StateHoliday'], axis =1)
    cols = 3
    rows = math.ceil(size / cols)
    fig, axes = plt.subplots(nrows= rows, ncols= cols)
    count = 0
    for i in stores:
        count = count + 1
        temp_store = view_frames[view_frames.Store == i]
        temp_store.index = temp_store['Date']
        if aggregation == 'sum':
            temp_store = temp_store.resample(group).sum()
        elif aggregation == 'mean':
            temp_store = temp_store.resample(group).mean()
        else:
            print "Not supported aggregation type: " + aggregation
            return
        plot_row_index = math.ceil(count / cols) - 1
        plot_col_index = count - plot_row_index * cols -1
        temp_store.plot(kind='line',x=temp_store.index,y='Sales',
            title = "Store_{0}".format(i), ax = axes[plot_row_index,plot_col_index])

def viewSalesData(data, group, size = 10):
    pass
