import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
import seaborn as sns

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
    rows = int(math.ceil(size / cols))
    rows = rows if rows > 0 else 0
    print("row: %d, col: %s" % (rows, cols))
    #fig, axes = plt.subplots(nrows = rows, ncols = cols)
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
            print("Not supported aggregation type: " + aggregation)
            return
        plot_row_index = int(math.ceil(count / cols) - 1)
        #plot_row_index = plot_row_index if plot_row_index > 0 else 0
        plot_col_index = int(count - plot_row_index * cols - 1)
        #print "plot row: %d, col: %d" % (plot_row_index, plot_col_index)
        #temp_store.plot(kind='line',x=temp_store.index,y='Sales',
        #    title = "Store_{0}".format(i), ax = axes[plot_row_index,plot_col_index])

def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)

def select_store(data, store_id, group, aggregation):
    data2 = data[data.Store == store_id]
    #print(data2)
    simple_data = data2.drop(['Store','Customers','Open','Promo','SchoolHoliday','StateHoliday'], axis =1)
    #print(simple_data.columns)
    simple_data.index = simple_data['Date']
    if aggregation == 'sum':
        simple_data = simple_data.resample(group).sum()
    elif aggregation == 'mean':
        simple_data = simple_data.resample(group).mean()
    else:
        print("Not supported aggregation type: " + aggregation)
        return
    simple_data.fillna(0)
    #reset store id
    store_number = np.empty(len(simple_data['Sales']))
    store_number.fill(store_id)
    simple_data['Store'] = store_number
    #print "selecting data for store: %d" % store_id
    #print simple_data
    return simple_data

def viewRandomStoreData(data, group, aggregation = 'sum',size = 10):
    stores = np.random.choice(len(data['Store'].unique()),size)
    return viewStoreData(data, stores, group, aggregation)

def viewStoreData(data, stores,group, aggregation = 'sum'):
    count = 0
    cols = 5
    plot_datas = pd.DataFrame({'Sales':[],'Date':[]})
    for i in stores:
        count = count + 1
        temp_store = select_store(data, i,group,aggregation)
        temp_store.reset_index(inplace = True)
        plot_datas = plot_datas.append(temp_store)
    plot_datas.sort_values("Date", ascending= True,inplace= True)
    g = sns.FacetGrid(plot_datas, col="Store", col_wrap = cols, size=2.5)
    g = g.map_dataframe(dateplot, "Date", "Sales")
    return stores
