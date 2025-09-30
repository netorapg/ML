
import pandas as pd
import numpy as np
import random


def test1():
    arr_1d = np.array([10, 20, 30, 40, 50])
    print("Before shuffle (1D):", arr_1d)
    np.random.shuffle(arr_1d)
    print("After shuffle (1D):", arr_1d)


def test2():
    ll = list(range(0,10))
    print(ll)
    random.shuffle(ll)
    print(ll)

test2()


def test_shuffle():
    fname = 'aula--jul-29/aaa.csv'
    data = pd.read_csv(fname, header=None)
    #print(len(data))
    ll = list(range(len(data)))
    #print(ll)
    random.shuffle(ll)
    #print(ll)

    print(data)
    print(data.iloc[ ll ])

test_shuffle()



def test_shuffle_v2():
    fname = 'aula--jul-29/aaa.csv'
    data = pd.read_csv(fname, header=None)
    ll = list(range(len(data)))
    random.shuffle(ll)
    # print(data)
    #print(data.iloc[ ll ])
    mydata = data.values.tolist()
    print(mydata)

#test_shuffle_v2()


def test_mascara():
    fname = 'cod-tmp/aaa.csv'
    data = pd.read_csv(fname, header=None)
    mascara = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    mascara = np.array( mascara, dtype=bool )

    #mascara[7] = True
    #mascara[15] = True
    #mascara[19] = True

    mascara[ [7,15,19] ] = True

    df = data[ mascara ]
    print(df)
