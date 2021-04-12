# -*- coding: utf-8 -*-
# === _trt1.py ===

# for experimentally usage

# embeded package
import os
import re
from copy import copy

# public package
import numpy as np
from pandas import DataFrame, read_csv
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpat

from tqdm import tqdm

# private package
# from pkmap import pkmap
from .pkmap_data import AD, pat_data, app_data
# from ekmapTK import KM


class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, ax, ax2, fig, interval):
        # self.lastind = 0

        # self.text = ax.text(0.05, 0.95, 'selected: none',
        #                     transform=ax.transAxes, va='top')
        # self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
        #                          color='yellow', visible=False)
        global data02

        self.ax = ax
        self.ax2 = ax2
        self.fig = fig
        self.cols = data02.columns
        self.inds = data02.index
        self.inds_ = ('Aggregate', 'Appliance1', 'Appliance2', 'Appliance3', 
                      'Appliance4', 'Appliance5', 'Appliance6', 
                      'Appliance7', 'Appliance8', 'Appliance9')
        self.ax20 = None

        if interval == 'hour':
            self.scope = 32
        elif interval == 'day':
            self.scope = 3
        elif interval == 'min':
            self.scope = 180
        self.noplot = True
        self.ConPatch = []

        print('\n\tclick the axis above to preview')


    def on_click(self, event):
        """
        docstring
        """
        print((event.xdata, event.ydata))
        if event.inaxes == self.ax:
            # print((event.xdata, event.ydata))
            # print((self.inds[int(event.ydata)], self.cols[int(event.xdata)]))
            
            self.ind_get = np.round(event.xdata).astype(np.uint32)
            self.y_get = np.round(event.ydata).astype(np.uint8)
            # print((self.ind_get, self.y_get))
            self.time_select = self.cols[self.ind_get]
            self.app_select = self.inds[self.y_get]
            self.app_select_ = self.inds_[self.y_get]
            
            self.refresh()


    def on_key(self, event):
        """
        docstring
        """
        if self.noplot:
            return None

        key = event.key
        if key == 'up':
            if self.y_get > 0:
                self.y_get -= 1
                self.app_select = self.inds[self.y_get]
                self.refresh()
        elif key == 'down':
            if self.y_get < self.inds.size:
                self.y_get += 1
                self.app_select = self.inds[self.y_get]
                self.refresh()
        elif key == 'left':
            self.ind_get -= self.scope
            self.ind_get = 0 if self.ind_get < 0 else self.ind_get
            self.time_select = self.cols[self.ind_get]
            self.refresh()
        elif key == 'right':
            self.ind_get += self.scope
            self.ind_get = self.olc.size if self.ind_get > self.cols.size else self.ind_get
            self.time_select = self.cols[self.ind_get]
            self.refresh()
        elif key == 'l':
            self.scope = np.round(self.scope * 1.6)
            print('\tamplify scope to {}'.format(self.scope))
            self.refresh()
        elif key in ('k', 's'):
            self.scope = np.round(self.scope / 1.6)
            print('\tshrink scope to {}'.format(self.scope))
            self.refresh()


    def add_patch(self, ind_low, ind_high):
        """
        docstring
        """

        self.ax.add_patch(
            mpat.Rectangle((ind_low, self.y_get-0.5), 
                            width=2*self.scope, height=1, 
                            color='c', fill=False, lw = 1)
        )
        
        # add cross axes patch (line)
        xy1 = ((ind_low, self.y_get+0.5), (ind_high, self.y_get+0.5))
        xy2 = ((self.xlim[0], self.ax2.get_ylim()[1]), 
               (self.xlim[1], self.ax2.get_ylim()[1]))
        self.ConPatch = [self.fig.add_artist(mpat.ConnectionPatch(
                                            xyA=xyA, coordsA=self.ax.transData, 
                                            xyB=xyB, coordsB=self.ax2.transData,)
                        ) for xyA, xyB in zip(xy1, xy2)]


    def refresh(self):
        """
        docstring
        """
         
        global data0

        # ind_get = DataFrame(range(self.cols.size))[self.cols==self.time_select].values[0,0]
        cell = self.cols.size
        ind_low = self.ind_get-self.scope if self.ind_get > self.scope else 0
        ind_high = self.ind_get+self.scope if self.ind_get+self.scope<cell else cell-1
        
        # a < b < c will be translated to (a<b) and (b<c) as a sugar 
        data_2plot = data0[((self.cols[ind_low] < data0.Time) & (data0.Time < self.cols[ind_high]))]
        # print(data_2plot)
        self.ax2.clear()
        self.ax2.plot(data_2plot['Aggregate'], 'gray', label='Aggregate')
        self.ax2.set_xlabel('date')
        self.ax2.set_ylabel('Aggregate')
        self.xlim = (data_2plot.index[0], data_2plot.index[-1])
        self.ax2.set_xlim(self.xlim)
        self.ax2.set_ylim(bottom=0)

        """
        plot data in x-axis
        """
        time_ticks = []
        time_labels = []
        last_date = ""
        last_date2 = ""
        for ind, t in data_2plot.Time.items():
            if t[11:13] == '00':
                if last_date != t[:10]:
                    last_date = t[:10]
                    time_ticks.append(ind)
                    time_labels.append(t[:10])
            elif t[11:13] == '12':
                if last_date2 != t[:10]:
                    last_date2 = t[:10]
                    time_ticks.append(ind)
                    time_labels.append('')

        time2x = DataFrame([time_ticks, time_labels]).T
        # print(time2x)
        self.ax2.set_xticks(time2x[0])
        self.ax2.set_xticklabels(time2x[1])

        if self.app_select != 'Aggregate':
            if self.ax20:
                self.fig.delaxes(self.ax20)
                self.ax20 = None
            self.ax20 = self.ax2.twinx()
            self.ax2.plot(range(3), 'm', label=self.app_select)
            self.ax20.plot(data_2plot[self.app_select_], 'm', )
            self.ax20.set_ylabel(self.app_select)
            self.ax20.set_ylim(bottom=0)

            if self.ax20.get_ylim()[1] > 0.4 * self.ax2.get_ylim()[1]:
                print(self.ax2.get_ylim()[1])
                self.ax20.set_ylim(top=self.ax2.get_ylim()[1])
        self.ax2.legend()
        
        [p.remove() for p in self.ax.patches]
        [p.remove() for p in self.ConPatch]
        plt.pause(1e-5)

        self.add_patch(ind_low, ind_high)
        self.fig.canvas.draw()
        self.noplot = False


def load1(self, house_number, interval:str='day'):
    """
    docstring
    """
    global data02, data0

    time_para = {
        # last_sec, time_mag, time_tail, time_cut, scope_default
        # start from Time='2013-11-28 12:15:35'
        'min':(0, 60, ':00', -3, ),
        'hour': (-15*60, 60*60, ':00:00', -6),
        'day': (-15*60-12*60*60, 60*60*24, ' 00:00:00', -9)
    }

    last_sec, time_mag, time_tail, time_cut = time_para[interval]
    
    data0 = self.data0
    file2save = 'house{}_by_{}.csv'.format(house_number, interval)
    path2save = '/'.join([self.file_dir, file2save])
    for file in os.scandir(self.file_dir):
        if file2save == file.name:
            # load exist file
            print('\tloading old data')
            data02 = read_csv(path2save, dtype={'Time':'str'})
            # print(data02)
            # print(data02.dtypes)
            return data02
    else:
        print('\t:counting {}'.format(file2save))
        # data02 = load1(house_number, interval='hr')
        # data02.to_csv(path2save, index=False)

    app_para = app_data[house_number]
    last_time = ""

    # print((last_sec, time_mag, time_tail, time_cut))
    ind = 0
    cols = ('Time', 'Aggregate', 'Appliance1', 'Appliance2', 'Appliance3', 
            'Appliance4', 'Appliance5', 'Appliance6', 
            'Appliance7', 'Appliance8', 'Appliance9')
    data02 = DataFrame(columns=cols)
    apps = {col:None for col in cols}
    for it in tqdm(data0.loc[:, cols].itertuples(index=False), 
                    total=self.len, ascii=False, leave=False): 
        new_time = it.Time[:time_cut]
        new_sec = int(it.Time[-2:])
        
        sec = new_sec - last_sec
        sec = sec if sec > 0 else sec + 60
        if new_time != last_time:
            if last_time:
                # do without first time
                data02.loc[ind, 'Time'] = last_time + time_tail
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
            # new_time == last_time
            apps = {col:apps[col]+k*sec for col, k in zip(cols[1:],  it[1:])}

        last_sec = new_sec
        # loop here

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


def plot_time(self, 
              house_number: int=6, 
              noax2: bool=False, 
              app_name = None,
              interval: str='day', 
              ):
    """
    docstring
    """
    global data02, data0

    # nomeralize `interval`
    if interval.lower() in ('m', 'min', 'mins', 'minite', 'minites'):
        interval = 'min'
    elif interval.lower() in ('h', 'hr', 'hrs', 'hour', 'hours'):
        interval = 'hour'
    elif interval.lower() in ('d', 'day', 'days'):
        interval = 'day'
    else:
        interval_default = 'day'
        Warning('\t:unrecognized interval: {}, use "{}" instead'.format(interval, interval_default))
        interval = interval_default
    
    data0 = self.data0
    data02 = load1(self, house_number=house_number, interval=interval)
    cols = tuple(data02.columns[1:])
    time = data02.Time
    data02 = DataFrame(data02.loc[:, cols].T, dtype='float_')
    # data02[data02<1] = None
    # print(data02)
    if app_name:
        cols = tuple(['Aggregate'] + list(app_name))
    
    data02.index = cols
    data02.columns = time
    # print(data02.index)
    # print(data02.columns)
    # print(data02.dtypes)

    """
    do plotting below
    """
    if noax2:
        fig, ax = plt.subplots(1,1,figsize=(15,6))
        ax2 = None      # eliminate unbound warnings
    else:
        fig, (ax, ax2) = plt.subplots(2,1,figsize=(15, 9))
    ax_1 = fig.add_axes([0.94, 0.5, 0.01, 0.4])
    # ax_1.set_visible = False
    plot1 = ax.imshow(data02, cmap='inferno', norm=LogNorm(), 
                    aspect='auto', interpolation='none', resample=False)
    ax.set_yticks(range(10))
    ax.set_yticklabels(cols)
    time2 = DataFrame([(ind,k) for ind, k in time.items() if k[11:13]=='00'])
    # print(time2)
    xrange = time2.reindex(np.linspace(0, len(time2)-1, 15, dtype='int_')).dropna()
    # xrange = xrange.dropna()
    # see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    # print(xrange)
    ax.set_xticks(xrange[0])
    ax.set_xticklabels([k[:10] if np.mod(n,2) else '' for k,n in zip(xrange[1], range(1,22)) ])
    ax.set_title('time analysis of House {} by {}'.format(house_number, interval), fontsize=20)
    fig.colorbar(plot1, cax = ax_1, label='by mean of each appliance')

    if not noax2:
        brower = PointBrowser(ax, ax2, fig, interval)
        fig.canvas.mpl_connect('button_release_event', brower.on_click)
        fig.canvas.mpl_connect('key_release_event', brower.on_key)

    # fig.tight_layout()
    # print(fig.set_dpi(200))
    plt.show()

    # end of plot_time()


if __name__ == "__main__":
    
    # plot_time(house_number=2, )

    pass
