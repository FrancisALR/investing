import datetime as dt
import matplotlib as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as web

style.use("ggplot")

start = dt.datetime(2000,1,1)
end = dt.datetime.now()

df = web.DataReader("V", "yahoo", start, end)
print(df.head())
