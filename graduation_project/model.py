import prepare_data

(X,y) = prepare_data.prepare_data_for_process()
print("prepare data complete")
print("================================================")

# 划分数据
train_ratio = 0.85
validate_ratio = 0.1
test_ratio = 0.05

num_records = len(X)
train_size = int(train_ratio * num_records)
validate_size = int(validate_ratio * num_records)
test_size = int(test_ratio * num_records)

test_index = train_size + validate_size

X_train = X[:train_size]
X_val = X[train_size:test_index]
X_test = X[test_index:]
y_train = y[:train_size]
y_val = y[train_size:test_index]
y_test = y[test_index:]
print("Data split completed...")
print("================================================")

# 评价指标
import pandas as pd
import numpy
from sklearn.metrics import mean_squared_error

# RMSPE
def score(y_act, y_pred):
    relative_err = numpy.square((y_act - y_pred) / y_act)
    return numpy.sqrt(numpy.sum(relative_err) / len(y_act))

# RMSE
def score_rmse(y_act, y_pred):
    return numpy.sqrt(numpy.mean((y_act - y_pred)**2))

def fit_estimator(X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import LinearSVR
    import random

#     lsvr = LinearSVR(random_state=store_id)
#   linear regression
    lsvr = LinearRegression()
    lsvr.fit(X, y)
    return lsvr
print("================================================")


# Overall linear regression models
x_df_train_all = pd.DataFrame(X_train)
x_df_test_all = pd.DataFrame(X_test)

all_estimator = fit_estimator(x_df_train_all.as_matrix(),pd.Series(y_train))
y_pred_all = all_estimator.predict(x_df_test_all.as_matrix())

result = score(pd.Series(y_test), pd.Series(y_pred_all))
print("Overall stores Linear regression - Testing error:")
print(result)
print("================================================")

# pre store linear regression model
def to_X_df(X, y):
    df = pd.DataFrame(X)
    df.columns = ['store_open', 'store_index', 'day_of_week', 'promo',
                      'year','month','day','state']
    df['Sales'] = pd.Series(y)
    return df

train_dic = dict(list(to_X_df(X_train, y_train).groupby('store_index')))
# print(train_dic[2].head(2))

test_dic = dict(list(to_X_df(X_test, y_test).groupby('store_index')))
# print(test_dic[2].head(2))

y_observed = []
y_predicts = []

for s_id in test_dic.keys():
    # Train data of Store
    train_of_store = train_dic[s_id]
    y_train_of_store = train_of_store['Sales']
    x_train_of_store = train_of_store.drop(['Sales','store_index'],axis =1)
    estimator = fit_estimator(x_train_of_store.as_matrix(),y_train_of_store)

    # Test data of Store
    test_of_store = test_dic[s_id]
    y_test_of_store = test_of_store['Sales']
    x_test_of_store = test_of_store.drop(['Sales','store_index'],axis =1)

    y_pred = estimator.predict(x_test_of_store)

    y_observed.extend(y_test_of_store)
    y_predicts.extend(y_pred)
#     print("Extended {} items".format(len(y_pred)))

result = score(pd.Series(y_observed), pd.Series(y_predicts))
print("Pre score Linear regression - Testing error:")
print(result)
print("================================================")

# Embedding network
import numpy
# 导入神经网络实现
from network import Enbedding_Network

def sample(X, y, n):
    '''random samples'''
    num_row = X.shape[0]
    indices = numpy.random.randint(num_row, size=n)
    return X[indices, :], y[indices]

# 可动态调整，这里选择20万个用于batch训练，并执行6次
batch_size = 200000
fit_epoch = 6

models = []

print("Build embedding network with Keras ...")
for i in range(5):
    X_train, y_train = sample(X_train, y_train, batch_size)  # take a random sample for training
    print("Number of samples used for training: " + str(y_train.shape[0]))
    model = Enbedding_Network(X_train, y_train, X_val, y_val, fit_epoch)
    model.fit_model(X_train, y_train, X_val, y_val)
    models.append(model)
print("================================================")

# Test set
# Evaluate model
def test_models(models, X, y):
    assert(min(y) > 0)
    predict_sales = numpy.array([model.predict(X) for model in models])
    predicted_sales_mean = predict_sales.mean(axis=0)
    relative_err = numpy.absolute((y - predicted_sales_mean) / y)
    result = numpy.sum(relative_err) / len(y)
    return result,predicted_sales_mean

print("Embedding network test set - Testing error rate ...")
r_val,predicted_sales_mean = test_models(models, X_test, y_test)
print(r_val)
print("================================================")

# print result
# 反序列化数据
import pickle
import numpy as np
import pandas as pd

# Restore label encoder for decoding
encoders = []
with open('les.pickle', 'rb') as le:
    les = pickle.load(le)
    encoders = les

# Prepare labels dataframe
test_df = pd.DataFrame(X_test)
test_df.columns = ['store_open', 'store_index', 'day_of_week', 'promo',
                  'year','month','day','state']

for idx,value in enumerate(test_df.columns):
    test_df[value] = encoders[idx].inverse_transform(test_df[value])

test_df.drop(['store_open','day_of_week','promo','state'], axis = 1, inplace = True)
test_df['Date'] = pd.to_datetime(test_df[['year','month','day']])
test_df.drop(['year','month','day'], axis = 1, inplace = True)
test_df['Sales'] = pd.Series(y_test)
# print(test_df.head(2))
# print(type(test_df['Date'][2]))
# print("")

# Prepare prediction dataframe
prd_df = pd.DataFrame({'Date':[],'Sales':[]})
prd_df['Date'] = test_df['Date']
prd_df['Sales'] = pd.Series(predicted_sales_mean)
# print(prd_df.head(2))
# print("")

# Prepare plot data
view_store_id = np.random.choice(len(test_df['store_index'].unique()),1)[0]
print("Picking Store {0} to plot:".format(str(view_store_id)))
plot_df = test_df[test_df.store_index == str(view_store_id)].copy()
plot_sales_df = prd_df.loc[plot_df.index]

plot_df.drop(['store_index'],axis = 1, inplace = True)
plot_df.set_index(['Date'], inplace=True)
plot_df.index.name=None
plot_sales_df.set_index(['Date'], inplace=True)
plot_sales_df.index.name=None

# print(plot_df.head(2))
# print(plot_sales_df.head(2))
print("done")

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
orig = plt.plot(plot_df, color='blue', label='Real Sales')
predicted = plt.plot(plot_sales_df, color='red', label='Predicted Sales')
plt.legend(loc='best')
plt.title('Store {0}: Real vs Predicted Sales'.format(str(view_store_id)))
plt.show()
