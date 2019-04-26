import argparse
import numpy as np
import pandas as pd
from importlib import import_module


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("surname")
    parser.add_argument("file")
    args = parser.parse_args(*argument_array)
    return args

def rmsq_loss(Y_predicted, Y):
    return (sum((Y_predicted - Y)**2)/len(Y))**0.5

if __name__ == "__main__":
    args = parse_args()
    name, surname = args.name, args.surname
    code = import_module("homework2_" + name + "_" + surname)
    df = pd.read_csv(args.file, index_col=0)
    
    X, Y = code.get_data(df)
    weights = code.fit_ridge_regression(X, Y)
    print(rmsq_loss(X.dot(weights), Y))
