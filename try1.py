# -*- coding: utf-8 -*-
# === _trt1.py ===

# for experimentally usage

import warnings
from pkmap import pkmap
from pkmap_data import AD, pat_data, app_data
# from ekmapTK import KM
import numpy as np
from pandas import DataFrame, read_csv

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tqdm import tqdm
from sklearn.cluster import KMeans
import os
import re


class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, ax, ax2, fig):
        # self.lastind = 0

        # self.text = ax.text(0.05, 0.95, 'selected: none',
        #                     transform=ax.transAxes, va='top')
        # self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
        #                          color='yellow', visible=False)
        global data02

        self.ax = ax
        self.ax2 = ax2
        self.fig = fig
        self.inds = data02.index
        self.cols = data02.columns
        self.ax20 = None


    def on_click(self, event):
        """
        docstring
        """

        if event.inaxes == self.ax:
            print((self.inds[int(event.ydata)], self.cols[int(event.xdata)]))
        self.app_select = self.inds[round(event.ydata)]
        self.time_select = self.cols[round(event.xdata)]
        print(f'{(self.app_select, self.time_select)=}')
        
        self.refresh()


    def refresh(self):
        """
        docstring
        """
         
        global data0

        # ind1 = data0.reindex(data0.Time==self.time_select)
        # ind2 = data0[data0.Time>self.time_select]
        ind_get = DataFrame(range(self.cols.size))[self.cols==self.time_select].values[0,0]
        # (self.cols)[self.cols==self.time_select][0]
        print(ind_get)
        cell = self.cols.size
        
        ind_low = ind_get-12 if ind_get > 12 else 0
        ind_high = ind_get+12 if ind_get+12<cell else cell-1
        print(f'{(ind_low, ind_high)=}')
        # a < b < c will be translated to (a<b) and (b<c) as a sugger 
        data_2plot = data0[((self.cols[ind_low] < data0.Time) & (data0.Time < self.cols[ind_high]))]
        # print(data_2plot)
        self.ax2.clear()
        self.ax2.plot(data_2plot['Aggregate'], 'gray', label='Aggregate')
        self.ax2.set_xlabel('data')
        self.ax2.set_ylabel('Aggregate')
        self.ax2.set_xlim(data_2plot.index[0], data_2plot.index[-1])
        self.ax2.set_ylim(bottom=0)
        
        if self.app_select != 'Aggregate':
            self.ax20 = self.ax2.twinx()
            self.ax2.plot(range(3), 'm', label=self.app_select)
            self.ax20.plot(data_2plot[self.app_select], 'm', )
            self.ax20.set_ylabel(self.app_select)
            self.ax20.set_ylim(bottom=0)
            
            if self.ax20.get_ylim()[1] > 0.4 * self.ax2.get_ylim()[1]:
                print(self.ax2.get_ylim()[1])
                self.ax20.set_ylim(top=self.ax2.get_ylim()[1])
        self.ax2.legend()
        
        self.fig.canvas.draw()
        if self.ax20:
            self.fig.delaxes(self.ax20)
            self.ax20 = None


def in_axes(event):
    """
    docstring
    """
    print(event)


def out_axes(event):
    """
    docstring
    """
    print(event)


def load1(house_number, interval:str='hr'):
    """
    docstring
    """
    global data02, data0

    p1 = pkmap('./REFIT/CLEAN_House' + str(house_number) + '.csv', count=False)
    data0 = p1.data0
    # pkmap.plot()
    # print(pkmap.data0)

    file2save = 'house'+ str(house_number) + '_in_hour.csv'
    path2save = './REFIT/' + file2save
    for file in os.scandir('./REFIT/'):
        if file2save == file.name:
            # load exist file
            print('\t:loading old data')
            data02 = read_csv(path2save, dtype={'Time':'str'})
            # print(data02)
            # print(data02.dtypes)
            return data02
            break
    else:
        print('\t:counting ' + file2save)
        # data02 = load1(house_number, interval='hr')
        # data02.to_csv(path2save, index=False)

    '''
    from 17'51" to 18'51"
    set 0'-9" for the first
    '''
    time_para = {
        # last_sec, time_mag, time_tail, time_cut
        # start from Time='2013-11-28 12:15:35'
        'min':(-9, 60, ':00', -3),
        'hr': (-9-15*60, 60*60, ':00:00', -6),
        'day': (-9-15*60-12*60*60, 60*60*24, ' 00:00:00', -9)
    }
    data = p1.data0
    app_para = app_data[house_number]
    last_time = ""

    if interval.lower() in ('m', 'min', 'mins', 'minite', 'minites'):
        last_sec, time_mag, time_tail, time_cut = time_para['min']
    elif interval.lower() in ('h', 'hr', 'hrs', 'hour', 'hours'):
        last_sec, time_mag, time_tail, time_cut = time_para['hr']
    elif interval.lower() in ('d', 'day', 'days'):
        last_sec, time_mag, time_tail, time_cut = time_para['day']
    else:
        last_sec, time_mag, time_tail, time_cut = time_para['min']
        Warning('\t:unrecognized interval: ' + interval + ', use "min" instead')
    print((last_sec, time_mag, time_tail, time_cut))
    ind = 0
    cols = ('Time', 'Aggregate', 'Appliance1', 'Appliance2', 'Appliance3', 
            'Appliance4', 'Appliance5', 'Appliance6', 
            'Appliance7', 'Appliance8', 'Appliance9')
    data02 = DataFrame(columns=cols)
    apps = {col:None for col in cols}
    first = True
    for it in tqdm(data.loc[:, cols].itertuples(index=False), 
                    total=p1.len, ascii=False, leave=False): 
        # print(it)

        # if first:
        #     # print(it)
        #     first = False
        new_time = it.Time[:time_cut]
        new_sec = int(it.Time[-2:])
        
        # ind = it.Index
        sec = new_sec - last_sec
        sec = sec if sec > 0 else sec + 60
        if new_time != last_time:
            if last_time:
                # sec +=  60
                # do without first time
                data02.loc[ind, 'Time'] = last_time + time_tail
                if first:
                    print(apps)
                    first = False
                for col in cols[1:]:
                    # if False and apps[col] < app_para[col]['thrd'] * time_mag:
                    if apps[col] < time_mag:
                        data02.loc[ind, col] = None
                    else:
                        mean = app_para[col]['mean'] * time_mag 
                        std = app_para[col]['std'] * time_mag
                        if std > 0:
                            data02.loc[ind, col] = (apps[col] / mean) 
                            # data02.loc[ind, col] = 
                        else:
                            data02.loc[ind, col] = None
                ind += 1
            # re-set
            last_time = new_time
            apps = {col:k*sec for col, k in zip(cols[1:], it[1:])}
            # print((last_time, apps))

        else:
            # sec = new_sec - last_sec
            apps = {col:apps[col]+k*sec for col, k in zip(cols[1:],  it[1:])}
        # if first:
        #     print(apps)
        last_sec = new_sec

    '''
    data02 is:
                   Time Aggregate Appliance1 ... Appliance4 Appliance5 ... Appliance9
    0  2013-11-28 12:15     15988          0 ...        420        796 ...        420
    1  2013-11-28 12:16     15887          0 ...        450        810 ...        420
    2  2013-11-28 12:17     15811          0 ...        434        811 ...        405
    ......
    '''
    data02.to_csv(path2save, index=False)

    return data02


def plot_time(house_number: int=6, noax2: bool=False):
    """
    docstring
    """
    global data02

    data02 = load1(house_number=house_number, interval='hr')
    cols = tuple(data02.columns)
    time = data02.Time
    data02 = DataFrame(data02.loc[:, cols[1:]].T, dtype='float_')
    # data02[data02<1] = None
    print(data02)
    # data02 = np.log(data02)
    data02.index = cols[1:]
    data02.columns = time
    print(data02.index)
    print(data02.columns)
    # print(data02.dtypes)

    # plotting below
    # fig = plt.figure(figsize=(12, 5))
    # ax = fig.add_axes( )
    if noax2:
        fig, ax = plt.subplots(1,1,figsize=(15,9))
    else:
        fig, (ax, ax2) = plt.subplots(2,1,figsize=(15, 9))
    ax_1 = fig.add_axes([0.94, 0.5, 0.01, 0.4])
    # ax_1.set_visible = False
    # plot1 = ax.pcolormesh(data02, cmap='inferno', norm=cLogNorm())
    plot1 = ax.imshow(data02, cmap='inferno', norm=LogNorm(), 
                    aspect='auto', interpolation='none', resample=False)
    # ax.set_xticklabels(time, rotation=45)
    ax.set_yticks(range(10))
    ax.set_yticklabels(cols[1:])
    time2 = DataFrame([(ind,k) for ind, k in time.items() if k[11:13]=='00'])
    # print(time2)
    # print(DataFrame([(ind,k) for ind, k in time.items() if k[5:10]=='04-01']))
    # print(np.linspace(0, len(time2)-1, 8, dtype='int_'))
    # print(time2.reindex(np.linspace(0, len(time2)-1, 8, dtype='int_'))[1])
    xrange = time2.reindex(np.linspace(0, len(time2)-1, 15, dtype='int_'))
    # xrange = time2.copy()
    xrange = xrange.dropna()
    # see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    print(xrange)
    ax.set_xticks(xrange[0])
    ax.set_xticklabels([k[:10] if np.mod(n,2) else '' for k,n in zip(xrange[1], range(1,22)) ])
    ax.set_title(r'time analysis of House ' + str(house_number) + ' by hour', fontsize=20)
    fig.colorbar(plot1, cax = ax_1, label=r'by mean of each appliance')
    
    if not noax2:
        brower = PointBrowser(ax, ax2, fig)
        fig.canvas.mpl_connect('button_release_event', brower.on_click)

    # fig.tight_layout()
    plt.show()


def do1():
    """
    docstring
    """

    len = 120
    X = np.random.rand(120, 10)
    data02 = DataFrame(X, index=['sfsf ' + str(k) for k in range(len)], 
            columns=['Aggregate'] + ['Appliance' + str(k+1) for k in range(9)])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))
    ax1.pcolor(data02.T, cmap='inferno')
    print((ax1.get_xbound(), ax1.get_ybound()))
    plt.show()


def do2(house_number=7):
    """
    generate mean and std of each app
    """

    p1 = pkmap('./REFIT/CLEAN_House' + str(house_number) + '.csv')
    data02 = p1.data0
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
        a[ind] = data02[col]
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
    plot_time(house_number=2, )
    # for ind in (0,):
    #     do2(ind+1)


