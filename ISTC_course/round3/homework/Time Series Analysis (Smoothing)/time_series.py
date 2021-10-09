import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime


def extrant_time_series_data():
    
    path = Path("data")

    time_series_data = list(path.iterdir())
    time_series_data
    
    # extranc the needed data
    england = pd.read_csv(time_series_data[0])
    TimeSeries = pd.read_csv(time_series_data[1])
    price_data = pd.read_csv(time_series_data[2])
    monthly_wage = pd.read_csv(time_series_data[3])
    energy_cons = pd.read_csv(time_series_data[4])
    
    england.Month = england.Month.apply(lambda x : datetime.strptime(x, "%Y-%m"))
    temp = england.iloc[:, 1:2]
    
    temp = [float(t) if not t.startswith('?') else float(t[1:]) 
            for t in temp.values[:, 0]]
    
    monthly_wage['Month'] = monthly_wage.iloc[:, 0].apply(lambda x: x[:x.find(";")])
    monthly_wage['RealWage'] = monthly_wage.iloc[:, 0].apply(lambda x: x[x.find(";") + 1:])
    del monthly_wage['Month;Real wage']
    
    monthly_wage.Month = monthly_wage.Month.apply(lambda x : datetime.strptime(x, "%Y-%m"))
    realwage = monthly_wage.iloc[:, 1:2]
    
    wage = [float(t) if not t.startswith('?') else float(t[1:]) 
            for t in realwage.values[:, 0]]
    
    energy_cons.Date = energy_cons.Date.apply(lambda x : datetime.strptime(x, "%Y-%m-%d"))
    
    return ((england, temp),
            (price_data), 
            (monthly_wage, wage), 
            (energy_cons, energy_cons['EnergyConsump']))