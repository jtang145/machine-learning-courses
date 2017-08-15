import pandas as pd
import numpy as np

def debug(str):
    print '------'
    print str
    print '------'

def read_data(file):
    data = pd.read_csv(file, dtype={"StateHoliday":np.str}, index_col="Date")

    return data
