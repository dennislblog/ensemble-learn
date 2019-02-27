
from multiprocessing import Pool
from Script.base import Ensemble
import pandas as pd
import os
import numpy as np
import sqlite3


def worker(namespace):
    work = Ensemble(X,y)
    file = "./Script/database/tmp_" + str(namespace) + ".db"
    try:
        work.init_db(file)
        work.gen_data_info()
        work.gen_model_info()
        work.gen_heuristic_info(max_stop=30)
    except:
        print("error worker = {}".format(namespace))

def get_result():
    path = "./Script/database"
    df=[]
    for fname in os.listdir(path):
        if fname.startswith("tmp"):
            db = sqlite3.connect(path + '/' + fname)
            with db:
                df.append(pd.read_sql_query("SELECT * FROM heuristic_info",db))
            db.close()
    tmp = pd.concat(df)
    tmp.to_sql("heuristic_info",sqlite3.connect("./Script/database/summary.db"))
    for fname in os.listdir(path):
        if fname.startswith("tmp"):
            os.remove(os.path.join(path, fname))

def load_data(file, separator=","):
    try:
        f = open(file, "r")
        s = [line for line in f]
        f.close()
    except:
        raise Exception
    s = filter(lambda e: e[0] != '@', s)
    s = [v.strip().split(separator) for v in s]
    df = np.array(s)
    X = np.asarray(df[:,:-1], dtype=float)
    d = {'positive': 1, 'negative': 0}
    y = np.asarray([d[v[-1].strip()] if v[-1].strip() in d else v[-1].strip() for v in s])
    return X, y


X,y = load_data(r'.\Data\ecoli-0-6-7_vs_5.dat')

if __name__ == '__main__':
    N_CORES = 8
    i = range(1,11)
    p = Pool(2*N_CORES)
    p.map(worker, i)
    get_result()