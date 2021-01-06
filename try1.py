# -*- coding: utf-8 -*-
# === _trt1.py ===

# for experimentally usage

from pkmap import pkmap
from pkmap_data import AD, pat_data
# from ekmapTK import KM
import numpy as np
from pandas import DataFrame

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def load1():
    """
    docstring
    """
    p1 = pkmap('./REFIT/CLEAN_House6.csv', count=False)
    # pkmap.plot()
    # print(pkmap.data0)

    '''
    from 17'51" to 18'51"
    set 0'-9" for the first
    '''

    data = p1.data0

    last_time = ""
    # -9    ->  each minite
    # -944  ->  each hour
    last_sec = -9 - 15*60
    ind = 0
    cols = ('Time', 'Aggregate', 'Appliance1', 'Appliance2', 'Appliance3', 
            'Appliance4', 'Appliance5', 'Appliance6', 
            'Appliance7', 'Appliance8', 'Appliance9')
    data2 = DataFrame(columns=cols)
    apps = {col:None for col in cols}
    first = True
    for it in data.loc[:, cols].itertuples(index=False): 
        # print(it)
        # [:-3] ->  each minite
        # [:-6] ->  each hour
        # [:-9] ->  each day
        # == warning: not starting from zero hour nor zero day! ==
        if first:
            print(it)
            first = False
        new_time = it.Time[:-6]
        new_sec = int(it.Time[-2:])
        time_tail = ':00:00'
        # ind = it.Index
        sec = new_sec - last_sec
        sec = sec if sec > 0 else sec + 60
        if new_time != last_time:
            if last_time:
                # sec +=  60
                # do without first time
                data2.loc[ind, 'Time'] = last_time + time_tail
                for col in cols[1:]:
                    data2.loc[ind, col] = apps[col]
                ind += 1
            # re-set
            last_time = new_time
            apps = {col:k*sec for col, k in zip(cols[1:], it[1:])}
            # print((last_time, apps))

        else:
            # sec = new_sec - last_sec
            apps = {col:apps[col]+k*sec for col, k in zip(cols[1:],  it[1:])}
        last_sec = new_sec

    '''
    data2 is:
                Time Aggregate Appliance1 ... Appliance4 Appliance5 ... Appliance9
    0  2013-11-28 12:15     15988          0 ...        420        796 ...        420
    1  2013-11-28 12:16     15887          0 ...        450        810 ...        420
    2  2013-11-28 12:17     15811          0 ...        434        811 ...        405
    ......
    '''

    return data2


def plot_time():
    """
    docstring
    """

    # data2 = load1()
    data2 = load1()
    cols = tuple(data2.columns)
    # print(p1.data0)
    # data3 = data2.loc[:, cols[1:]].T
    time = data2.Time
    data2 = DataFrame(data2.loc[:, cols[1:]].T, dtype='float_')
    print(data2)
    data2[data2<100] = None
    data2 = np.log(data2)
    data2.index = cols[1:]
    data2.columns = time
    print(data2.index)
    print(data2.columns)
    # print(data2.dtypes)

    # plotting below
    # fig = plt.figure(figsize=(12, 5))
    # ax = fig.add_axes( )
    fig, ax = plt.subplots(1,1,figsize=(12, 5))
    ax.pcolor(data2, cmap='inferno')
    # ax.set_xticklabels(time, rotation=45)
    ax.set_yticks(range(10))
    ax.set_yticklabels(cols[1:])
    plt.show()


def do1():
    """
    docstring
    """

    len = 120
    X = np.random.rand(120, 10)
    data2 = DataFrame(X, index=['sfsf ' + str(k) for k in range(len)], 
            columns=['Aggregate'] + ['Appliance' + str(k+1) for k in range(9)])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))
    ax1.pcolor(data2.T, cmap='inferno')
    print((ax1.get_xbound(), ax1.get_ybound()))
    plt.show()


def do2(house_number=7):
    """
    generate mean and std of each app
    """

    p1 = pkmap('./REFIT/CLEAN_House' + str(house_number) + '.csv')
    data2 = p1.data0
    print(p1.app_name)

    with open('dd', 'a') as f:
        f.write('# House ' + str(house_number) + '\n')
        f.write(str(house_number) + ':{\n')
    a = list(range(10))
    k = list(range(10))
    cols = ('Aggregate', 'Appliance1', 'Appliance2', 'Appliance3', 
            'Appliance4', 'Appliance5', 'Appliance6', 
            'Appliance7', 'Appliance8', 'Appliance9')
    for ind, col in enumerate(cols):
        a[ind] = data2[col]
        k[ind] = KMeans(n_clusters=2).fit(np.array(a[ind]).reshape(-1,1))
        print(f'{k[ind].cluster_centers_}')
        if np.diff(k[ind].cluster_centers_.reshape(1,-1))[0][0]>1e-5:
            isoff = a[ind][k[ind].labels_<1]
            ison = a[ind][k[ind].labels_>0]
        elif np.diff(k[ind].cluster_centers_.reshape(1,-1))[0][0]<-1e-5:
            isoff = a[ind][k[ind].labels_>0]
            ison = a[ind][k[ind].labels_<1]
        else:
            # cluster_centers_ may be [[0.], [0.]]
            with open('dd', 'a') as f:
                f.write('\t"' + col + '":{')
                if col != 'Aggregate':
                    f.write('\n\t\t"name": "' + p1.app_name[ind-1] + '",')
                f.write('\n\t\t"thrd": 0')
                f.write(',\n\t\t"mean": 0.0')
                f.write(',\n\t\t"std": 0.0')
                f.write(',\n\t},\n')
            continue

        max = isoff.max()
        min = ison.min()
        thrd = int((max+min)/2)
        print(thrd)

        mean = ison.mean()
        std = ison.std()
        with open('dd', 'a') as f:
            f.write('\t"' + col + '":{')
            if col != 'Aggregate':
                f.write('\n\t\t"name": "' + p1.app_name[ind-1] + '",')
            f.write('\n\t\t"thrd": ' + str(thrd))
            f.write(',\n\t\t"mean": ' + str(mean))
            f.write(',\n\t\t"std": ' + str(std))
            f.write(',\n\t},\n')
    with open('dd', 'a') as f:
        f.write('},\n')

    return None


if __name__ == "__main__":
    
    # do1()
    plot_time()
    # for ind in (0,):
    #     do2(ind+1)


