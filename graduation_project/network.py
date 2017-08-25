import pickle
import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Reshape
from keras.layers.embeddings import Embedding

def split_features(X):
    X_list = []

    store_index = X[..., [1]]
    X_list.append(store_index)

    day_of_week = X[..., [2]]
    X_list.append(day_of_week)

    promo = X[..., [3]]
    X_list.append(promo)

    year = X[..., [4]]
    X_list.append(year)

    month = X[..., [5]]
    X_list.append(month)

    day = X[..., [6]]
    X_list.append(day)

    State = X[..., [7]]
    X_list.append(State)

    return X_list

class Model(object):

    def evaluate(self, X_val, y_val):
        assert(min(y_val) > 0)
        return self.model.evaluate(self.preprocessing(X_val), self._val_for_fit(y_val),batch_size = 64)

class Enbedding_Network(Model):
    def __init__(self, X_train, y_train, X_val, y_val, epoch = 5):
        super().__init__()
        self.epoch = epoch
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__init_keras_model('mean_absolute_error')

    def __init_keras_model(self, loss='mean_absolute_error'):
        models = []

        model_store = Sequential()
        model_store.add(Embedding(1115, 10, input_length=1))
        model_store.add(Reshape(target_shape=(10,)))
        models.append(model_store)

        model_dow = Sequential()
        model_dow.add(Embedding(7, 6, input_length=1))
        model_dow.add(Reshape(target_shape=(6,)))
        models.append(model_dow)

        model_promo = Sequential()
        model_promo.add(Dense(1, input_dim=1))
        models.append(model_promo)

        model_year = Sequential()
        model_year.add(Embedding(3, 2, input_length=1))
        model_year.add(Reshape(target_shape=(2,)))
        models.append(model_year)

        model_month = Sequential()
        model_month.add(Embedding(12, 6, input_length=1))
        model_month.add(Reshape(target_shape=(6,)))
        models.append(model_month)

        model_day = Sequential()
        model_day.add(Embedding(31, 10, input_length=1))
        model_day.add(Reshape(target_shape=(10,)))
        models.append(model_day)

        model_germanstate = Sequential()
        model_germanstate.add(Embedding(12, 6, input_length=1))
        model_germanstate.add(Reshape(target_shape=(6,)))
        models.append(model_germanstate)

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        # TODO: RMSPE implementation
        self.model.compile(loss='mean_absolute_error', optimizer='adam')


    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def fit_model(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       nb_epoch=self.epoch, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        print("\r Error rate for validation dataset: ", self.evaluate(X_val, y_val))
        print("")


    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def predict(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
