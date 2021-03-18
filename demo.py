
from typing import Iterable
import warnings

from pandas.io.parsers import read_csv
warnings.filterwarnings("ignore")
import os, re
from multiprocessing import Pool

from copy import copy
from time import time, sleep
from collections import OrderedDict
from nilmtk import DataSet
# from nilmtk.utils import print_dict
# from nilmtk.api import API
# from nilmtk.dataset_converters import convert_greend
# from nilmtk.disaggregate import Mean

# from nilmtk_contrib.disaggregate import afhmm as converter

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from numpy import linspace
from numpy import array as narray
from numpy import log, sqrt, diff
from numpy import append as nappend

from pandas import read_csv

from pkmap import PKMap
from pkmap import read_REFIT, gen_PKMap
from pkmap.PKM import Hellinger
from pkmap.utils import do_plot_single2 as do_plot
from pkmap.utils import NoPrints, beauty_time
# from pkmap.house_preview import plot_time
# data = DataSet('./__data/iawe.h5')


def f(args):
    ind, dt = args
    # print('get {} as ind in f()'.format(ind))

    return ind, dt

# path = r'D:\dataset\REFIT\CLEAN_REFIT_081116\CLEAN_House1.csv'

# for dirpath, dirname, files in os.walk(os.path.dirname(path)):
#     for filename in files:
#         if filename.endswith('.csv'):
#             pm = pkmap(os.path.join(dirpath, filename), slice = 12)
#             pm.plot(fig_types='png', do_show=False)

# print(pm)

# data2, data0 = read_REFIT(path)

'''
test working with nilmtk
'''

# D1 = DataSet(r'D:\dataset\REFIT\refit.h5')
# D1 = DataSet(r'D:\dataset\iawe\iawe.h5')
# D1 = DataSet(r'D:\dataset\REDD\redd.h5')
# meters = D1.buildings[1].elec.submeters().meters

# avail_ac = [m.get('type') for m in meters[0].device['measurements']
#          if m['physical_quantity']=='power']

# m1 = [next(m.load()) for m in meters]
# m2 = pd.concat(m1, axis=1)
# m2.columns = [m.appliances[0].identifier.type for m in meters]
# print(m2)
# data2 = pkmap(D1.buildings[12])
# do_plot(data2)

'''
log_file = './record4.csv'
with open(log_file, 'a') as f:
    f.write('House,Appliance,BM\n')
for nb in range(21):
# for nb in (11,  ):
    nb += 1
    # print(nb)
    p1 = PKMap(D1.buildings[nb],) 
    p1.plot(fig_types='d', no_show=True)
    for np in range(9):
        np += 1
        # p1.plot(fig_types=('svg', 'png'))

        bm = p1.BM(obj=np, no_plot=True, fig_types='d')
        with open(log_file, 'a') as f:
            f.write('{},{},{}\n'.format(p1.house_name, np, bm))
'''

def f2(arg):
    ind, day, d0 = arg
    datat = d0[d0.index.date == day]
    if datat.index[0].hour < 2 and datat.index[-1].hour > 21:
        print(day)
        return ind, datat
    else:
        return ind, None


def do3(app_nums: Iterable=(5,), house_n: int=1, 
        no_save: bool=False, no_load: bool=True, 
        ):
    '''

    '''

    global D1
    if not 'D1' in dir():
        # if `D1` has not defined
        D1 = DataSet(r'D:\dataset\REFIT\refit.h5')
    p1 = PKMap(D1.buildings[house_n], no_count=True, no_load=no_load)

    # no_old = OrderedDict.fromkeys(app_nums)      # not get loaded data
    is_old = {k:False for k in app_nums}     # not get loaded data
    bm0 = {k:None for k in app_nums}
    sbm0 = {k:None for k in app_nums}
    BMs = {k:[] for k in app_nums}
    D0s = {k:[] for k in app_nums}
    D1s = {k:[] for k in app_nums}
    dates = []

    # re-use data by caching
    cache_dir = os.path.join(os.getcwd(), 'BMcache')
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if not no_save:
        cache_name0 = '_'.join(['BMcache', p1.dataset, p1.house_name, ''])
        for an in app_nums:
            cache_name = ''.join([cache_name0, p1.ins_name['active'][an-1], '.csv'])
            for dirp, dirn, files in os.walk(cache_dir):
                # print('dirpath is {}'.format(dirp))
                # print('dirname is {}'.format(dirn))
                # print('have {} files'.format(len(files)))
                files_l = narray([x.lower() for x in files])
                cache_l = cache_name.lower()
                if cache_l in files_l:
                    cache_name2 = narray(files)[files_l == cache_l][0]
                    with open(os.path.join(cache_dir, cache_name2), 'r') as f:
                        data_old = read_csv(f, index_col=0, header=0, 
                                            names=('D0', 'D1', 'bm'), 
                                            parse_dates=True, 
                                            )
                    D0s[an] = data_old.D0
                    D1s[an] = data_old.D1
                    BMs[an] = data_old.bm
                    is_old[an] = True
                    break
            else:
                is_old[an] = False
                print('cache_name is {}'.format(cache_name), ' '*16)
    else:
        cache_name0 = ""

    # init
    print('initializing ...', ' '*64, end='\r')
    if not isinstance(app_nums, Iterable):
        app_nums = (app_nums, )
    with NoPrints():
        for an in app_nums:
            # an = int(an)
            bm0[an], sbm0[an] = p1.BM(obj=an, sel_ac='active', no_plot=True)
    p1.appQ['dd'] = 9
    p1.app_name['dd'] = p1.app_name['active']
    p1.ins_name['dd'] = p1.ins_name['active']
    datas = []

    d0 = p1.data0['active']
    # if not any([is_old[an] for an in app_nums]):
    if all([is_old[an] for an in app_nums]):
        print('{0} all `{1}` App exist! {0}'.format('='*6, len(app_nums)))
        years = ()
    else:
        print('app {} not exist'.format(', '.join([str(an) for an in app_nums if not is_old[an]])))
        years = set(d0.index.year)

    sleep(1)
    t0 = time()
    app_rems = [an for an in app_nums if not is_old[an]]
    for _year in years:
    # for _year in (2014, ):
        print('\r\t\tcutting ... ', ' '*8, end='\r')
        # the hash of int is neurally in order
        d0_y = d0.loc[d0.index.year==_year]
        months = set(d0_y.index.month)
        for _month in months:
        # for _month in (9, ):
            d0_m = d0_y.loc[d0_y.index.month==_month]
            days = set(d0_m.index.day)
            days = (1, )
            for _day in days:
            # for _day in range(5, 11):
                # d0_x = d0_m.loc[d0_m.index.day==_day]
                d0_x = d0_m.copy()
                # dates.append('-'.join(narray([_year, _month, _day], dtype=str)))
                dates.append('-'.join(narray([_year, _month], dtype=str)))

                # set `end` to steady the cursor
                print('\r\t\treading {}'.format(dates[-1]), end='')
                # if d0_x.index[0].hour < 1 and d0_x.index[-1].hour > 22:
                if True:
                    datas.append(d0_x)
                    with NoPrints():
                        p1.data0['dd'] = d0_x.copy()
                        p1.data2['dd'] = gen_PKMap(p1, data0=d0_x, key='dd')
                        for an in app_rems:
                            bm, sbm = p1.BM(obj=an, sel_ac='dd', no_plot=False, 
                                            no_show=True, fig_types=('ty.jpg',))
                            BMs[an].append(bm)
                            D1s[an].append(Hellinger(sbm, sbm0[an]))
                            D0s[an].append(Hellinger(sbm))

                else:
                    # imcomplete day, discard
                    d0id0, d0id1 = d0_x.index[0], d0_x.index[-1]
                    print('\tfind invalid time: {}, {:0>2}:{:0>2} to {:0>2}:{:0>2}'.format(
                        d0id0.date(), d0id0.hour, d0id0.minute, d0id1.hour, d0id1.minute
                        ), end='')
        del d0_y, d0_m, d0_x

    # ring to alart the finishing
    if len(app_rems) < 2:
        print('\r\a `{}` app in {} of {} days cost {} '.format(
                len(app_rems), len(datas),
                len(set(d0.index.date)), beauty_time(time()-t0)+' '*128))
    else:
        print('\r\a `{}` apps in {} of {} days cost {} '.format(
                len(app_rems), len(datas),
                len(set(d0.index.date)), beauty_time(time()-t0)+' '*128))

    if datas:
        d1 = pd.concat(datas, )
    
    # plt.plot(d1.index, range(len(d1.index)))
    # plt.show()
    # sleep(1e-15)
    
    for an in app_nums:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(121)
        vio1 = ax.violinplot([D0s[an], D1s[an]], )
        h0 = Hellinger(sbm0[an])
        xlim = ax.get_xlim()
        print(xlim)
        ax.plot(xlim, [h0, h0], color='cornflowerblue')
        ax.plot(xlim, [0, 0], 'k')
        ax.set_xlim(xlim)
        ax.set_xticks([1,2])
        ax.set_xticklabels(['$D_0$', '$D_1$'])
        ylim = ax.get_ylim()
        tick = [k for k in ax.get_yticks() if k>ylim[0] and k<ylim[1]]
        ax.set_yticks(nappend(tick, h0))
        ntick = ['{:.1f}'.format(k) for k in tick]
        ax.set_yticklabels(nappend(ntick, r'$D_{pH}$'+' is{0}\n{1:.2f}{0}'.format(' '*9, h0)))
        plt.grid()

        ax2 = fig.add_subplot(122)
        vio2 = ax2.violinplot(BMs[an], )
        # vio2['bodies'][0].set_facecolor('r')
        # vio2['bodies'][0].set_edgecolor('violet')
        xlim2 = ax2.get_xlim()
        print(xlim2)
        ax2.plot(xlim2, [bm0[an], bm0[an]], 'b')
        ax2.set_xlim(xlim2)
        ylim2 = ax2.get_ylim()
        ax2.set_ylim(top=1.06, bottom=0)
        ax2.yaxis.tick_right()
        ax2.set_xticks([1])
        ax2.set_xticklabels(['BM'])
        tick = [k for k in ax2.get_yticks() if k < 1.1]
        ax2.set_yticks(nappend(tick, bm0[an]))
        print(ax2.get_yticks())
        ntick = ['{:.1f}'.format(k) for k in tick]
        ax2.set_yticklabels(nappend(ntick, '{0}BM is\n{0}{1:.3f}'.format(' '*7, bm0[an])))
        plt.grid()
        
        plt.savefig('./figs/D(bm)_{}_{}_A{}_{}.png'.format(
                    p1.dataset, p1.house_name, an, p1.ins_name['active'][an-1]
                    ), bbox_inches='tight')
        # plt.show()
        
        # save data
        t0 = time()
        if not no_save and not is_old[an]:
            cache_name = ''.join([cache_name0, p1.ins_name['active'][an-1], '.csv'])
            print('='*6, 'saving into `{}`'.format(cache_name), '='*6, end='\r')
            with open(os.path.join(cache_dir, cache_name), 'w') as f:
                f.write('date, D0, D1, BM')
                for _date, _d0, _d1, _bm in zip(dates, D0s[an], D1s[an], BMs[an]):
                    f.write('\n')
                    f.write(','.join(narray([_date, _d0, _d1, _bm], dtype=str)))

            print('='*6, 'done saving'.format(beauty_time(time()-t0)))

    pass
'''
test func: plot_time
'''
# pm = PKMap(file=path, no_count=True)
# pm.preview()


'''
something
'''


if __name__ == "__main__":
    print('# `try2` run as main')
    # with Pool(processes=3) as P:
    #     # for ind, dt in enumerate(['a', 'b', 'c', 'd', 'e']):
    #     #     res = P.map(do2, ((ind, dt), ))
    #     #     print('in for loop {}'.format(ind))
    #     res = P.map(do2, ((ind, dt) for ind, dt in enumerate(['a', 'b', 'c', 'd', 'e'])))
    # print(res)
    
    [do3((2,5,), house_n=k) for k in range(1, 22) if k ==1]
    # do3((1, ))

else:
    print('# `try2` run as: {} but main'.format(__name__))