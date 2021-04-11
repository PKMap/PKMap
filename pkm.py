# -*- coding: utf-8 -*-
# === pkm.py ===

import os
from re import findall, match
from copy import copy
from collections.abc import Iterable
from typing import Dict 
from time import time

import matplotlib.pyplot as plt
from nilmtk import dataset
from numpy.lib.arraysetops import isin
from numpy.lib.npyio import save
from pandas import DataFrame 
from pandas import read_excel
from pandas import concat

from numpy import diff, array, log, log10
from numpy import abs, sqrt
from numpy import nansum as sum
from numpy import nan, isnan
from sklearn.cluster import KMeans

from tqdm import tqdm

try:
    from nilmtk.building import Building
    from nilmtk import DataSet
    no_nilmtk = False
except ModuleNotFoundError:
    print('no `nilmtk` found')
    Building = type(None)
    DataSet = type(None)
    no_nilmtk = True
except Exception as E:
    print('{} happens while importing nilmtk'.format(E))
    no_nilmtk = True

from .utils import gen_PKMap2 as gen_PKMap
from .utils import read_REFIT2 as read_REFIT
from .utils import do_plot, do_plot_BM
from .utils import beauty_time, NoPrints
from .house_preview import plot_time


def Hellinger(P: dict, Q: dict=None, no_check: bool=False):
    '''
    caculate the pseudo-Hellinger distance

    P, Q: the SBMs, a dict of float in [-1,1]

    return: a float
    '''

    if Q is None:
        Q = {k:0 for k in P.keys()}
    
    if not no_check:
        Q = {k:Q[k] if Q[k] else 0 for k in Q.keys()}
        P = {k:P[k] if P[k] else 0 for k in P.keys()}
    
    # p & q ought be synchronous as they share same keys()
    p = array([sqrt(x) if x > 0 else 0-sqrt(0-x) for x in P.values()])
    q = array([sqrt(x) if x > 0 else 0-sqrt(0-x) for x in Q.values()])

    d = 0.5*sum((p-q)**2)
    # if sum(p-q)>0:
    #     return sqrt(d)
    # else:
    #     return 0-sqrt(d)
    return sqrt(d)


def HellingerO(P: dict, Q: dict=None):
    '''
    caculate the pseudo-Hellinger distance
    in `original` type

    P, Q: the SBMs, a dict of float in [-1,1]

    return: a float
    '''

    if Q is None:
        Q = {k:0 for k in P.keys()}
    
    # p & q ought be synchronous as they share same keys()
    p = array([sqrt(0-x) if x < 0 else sqrt(x) for x in P.values()])
    q = array([sqrt(0-x) if x < 0 else sqrt(x) for x in Q.values()])

    d = 0.5*sum((p-q)**2)
    return sqrt(d)


class PKMap(object):
    """
    docstring
    """
    
    def __init__(self, file = None, 
                 model: str='thrd', 
                 n_slice=None, 
                 sample_period: int=None,
                 no_count: bool=False,
                 no_load: bool=False,
                 ):
        """
        docstring

        file: .csv file path | nilmtk.Building
            input a data file and gogogo

        !!! do not use `no_load`
        """

        if file is None:
            # None(s) share same id
            print('file is None!')
            # here is a default test file
            self.file = './REFIT/CLEAN_House7.csv'
        else:
            self.file = file
        self.model = model
        self.n_slice = n_slice
        self.no_count = no_count
        self.no_load = no_load
        self.sample_period = sample_period
        self.cache_dir = os.path.join(os.getcwd(), 'cache')
        self.bm = {}
        self.Dph = {}

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        if isinstance(self.file, str) and self.file.endswith('.csv'):
            # 
            self.dataset = 'REFIT'
            _key = 'active'
            self.file_dir = os.path.dirname(self.file)
            self.house_name = findall(r'House\d+', self.file)[-1]
            self.house_number = int(findall(r'\d+', self.house_name)[-1])
            app_name = read_excel(os.path.join(self.file_dir, 'MetaData_Tables.xlsx'), 
                        sheet_name=''.join(['House ', str(self.house_number)]), 
                        usecols=('Aggregate', ), ).values[:]
            # cleaning as having '???' inside
            self.app_name[_key] = tuple(['Unknown' if match(r'\?+', n) else n.replace(' ', '_')
                            for n in app_name.reshape(1,-1)[0]])
            # using as name_app[n][0]
            self.appQ[_key] = len(self.app_name)
            self.load(no_count)
            self.data0 = {_key: self.data0}
            self.data2 = {_key: self.data2}
            # self.len = len(self.data0.index)
            self.isnilmtk = False

        elif isinstance(self.file, str) and file.endswith('.h5'):
            # TODO
            print('======  get .h5 file!  ======')
            if no_nilmtk:
                raise ModuleNotFoundError("no 'nilmtk' to read .h5 file")
            
            self.file_dir = None
            self.house_name = None
            self.house_number = None
            self.app_name = ()
            self.appQ = len(self.app_name)
            # self.len = len(self.data0.index)

            data = DataSet(self.file)

            for ind in data.buildings.keys():
                # TODO: manage in multiple houses
                pass
            self.isnilmtk = True


        elif isinstance(self.file, Building):
            # successfully import nilmtk
            print('======  get `nilmtk.Building`!  ======')
            self.dataset = self.file.metadata['dataset']
            self.file_dir = None
            self.house_number = self.file.metadata['instance']
            self.house_name = self.file.metadata['original_name']

            self.load_nilm(model=self.model, 
                           n_slice=self.n_slice, no_count=self.no_count, 
                           no_load=self.no_load)
            self.isnilmtk = True
            
        else:

            print('====== get unknow type: {}! ======'.format(type(self.file)))

    def load_nilm(self, 
                  model: str='thrd', 
                  n_slice=None, 
                  no_count: bool=True, 
                  no_load: bool=False,
                  no_save: bool=False,
                  ):
        """
        docstring
        """
        # data2: pseudo truth table
        # data0: original csv data
        meters = self.file.elec.submeters().meters
        self.data0 = {}
        self.data2 = {}
        self.app_name = {}
        self.ins_name = {}
        self.appQ = {}
        self.avail_key = set([])
        if 'power' in meters[0].available_physical_quantities():
            avail_ac = set()
            for meterx in meters:
                avail_ac |= {m.get('type') for m in meters[0].device['measurements']
                            if m['physical_quantity']=='power'}
            print('find available_ac: {}'.format(avail_ac))
            self.avail_ac = avail_ac
            # [print(m.available_ac_types('power')) for m in meters]

            # loading data
            main_ = next(self.file.elec.meters[0].load())
            st_day = main_.index[0].date()
            ed_day = main_.index[-1].date()
            m2 = None
            for ac1 in avail_ac:
                t0 = time()
                print('\tloading `{}` data from {}, '.format(ac1, self.house_name), end=' '*42)
                lst = [42, ]
                words = []
                if self.sample_period:
                    m1 = [words.append('with column `{}`'.format(m.appliances[0].label(pretty=True))) 
                        or print('\b'*lst[-1] + words[-1], end=' '*16+'\b'*16)
                        or lst.append(len(words[-1]))
                        or next(m.load(ac_type=ac1, sample_period=self.sample_period)).loc[st_day:ed_day] for m in meters 
                        if ac1 in m.available_ac_types('power')]
                else:
                    m1 = [words.append('with column `{}`'.format(m.appliances[0].label(pretty=True))) 
                        or print('\b'*lst[-1] + words[-1], end=' '*16+'\b'*16)
                        or lst.append(len(words[-1]))
                        or next(m.load(ac_type=ac1)).loc[st_day:ed_day] for m in meters 
                        if ac1 in m.available_ac_types('power')]

                m2 = concat(m1, axis=1)     # combine each app to one DataFrame

                app_names = [(m.appliances[0].identifier.type).title().replace(' ', '')
                            for m in meters if ac1 in m.available_ac_types('power')]
                self.app_name[ac1] = app_names
                print('\rfinish loading `{}` cols in {}'.format(len(app_names), beauty_time(time()-t0)) +' '*64)


                ins_names = set([])
                self.ins_name[ac1] = []
                for name in app_names:
                    if name in ins_names:
                        # duplicate, add a suffix (2,3,4,...)
                        for suf in range(2, 9):
                            name2 = ''.join([name, str(suf)])
                            if not name2 in ins_names:
                                # reach a new name, load and quit
                                ins_names.add(name2)
                                self.ins_name[ac1].append(name2)
                                break
                    else:
                        ins_names.add(name)
                        self.ins_name[ac1].append(name)
                self.ins_name[ac1] = tuple(self.ins_name[ac1])
                # ins_names = tuple([m.appliances[0].label(pretty=True) for m in meters
                                    # if ac1 in m.available_ac_types('power')])
                m2.columns = self.ins_name[ac1]
                self.data0[ac1] = m2
                self.appQ[ac1] = len(m2.columns)
                if not no_count:
                    print(m2)
                    self.data2[ac1] = gen_PKMap(self, key=ac1, 
                                                model=model, n_slice=n_slice,
                                                no_save=no_save)
                    self.avail_key.add(ac1)
                
            self.len = len(m2.index)
        elif 'energy' in meters[0].available_physical_quantities():
            print('energy type found in {}'.format(self.file.metadata['dataset']))

        else:
            raise('unknown physical_quantities: {}'.format(meters[0].available_physical_quantities()))

        # [print(datax) for datax in self.data0.values()]
        self.PTb = self.data2

        return None


    def load(self, no_count=False):
        """
        docstring
        """
        # data2: pseudo truth table
        # data0: original csv data
        print('\t run `read_REFIT`! ')
        
        read_REFIT(self, no_count=no_count)
        self.PTb = self.data2
        
        return None


    def plot(self, data2=None, key:str='active',
            cmap:str='inferno_r', fig_types=(), 
            no_show:bool=False, 
            titles="", pats=[], 
            **args):
        """
        docstring

        !!!
            caution when offering `data2`
        !!!
        """

        # data_tp is always a dict
        if key.lower() in ('all', 'a', 'as'):
            print('plotting all ac_type: {}'.format(self.avail_ac))
            for key_ in self.avail_key:
                # print('plotting: {}'.format(data2))
                do_plot(self, data2=None, key=key_, 
                        cmap=cmap, fig_types=fig_types, no_show=no_show,
                        titles=titles, pats=pats, **args)
        else:
            # fix input
            if not key in self.avail_key:
                print('key = `{}` is not available!'.format(key))
                key = list(self.avail_ac)[0]
            print('plotting default ac_type as {}'.format(key))

            if data2 is None:
                if not self.data2:
                    if self.isnilmtk:
                        # self.load_nilm(model=self.model, n_slice=self.n_slice, no_count=False)
                        for ac1 in self.avail_ac:
                            self.data2[ac1] = gen_PKMap(self, key=ac1, model=self.model, n_slice=self.n_slice)
                            self.avail_key.add(ac1)
                    else:
                        self.load(no_count=False)
                data2 = self.data2[key]
            else:
                pass

            do_plot(self, data2=data2, key=key, 
                    cmap=cmap, fig_types=fig_types, 
                    no_show=no_show,
                    titles=titles, pats=pats, **args)

        return None


    def preview(self, **args):
        """
        docstring

        not working well by now
        """
        # TODO: add violin plot on 9 apps of each House
        # to check if multi-apps are recorded in a single channel

        plot_time(self, house_number=self.house_number, 
                  app_name=self.app_name,
                  **args)

        return None


    def generate(self, save_name: str='mean&std_ON'):
        """
        generate mean and std of each app

        save_name: file name to write
                    (will be rewrited)

        return: None
        """
        with open(save_name, 'w') as f:
            f.write('')

        with open(save_name, 'a') as f:
            f.write('# House {}\n'.format(self.house_number))
            f.write('{}:{\n'.format(self.house_number))
        a = list(range(10))
        k = list(range(10))
        cols = ('Aggregate', 'Appliance1', 'Appliance2', 'Appliance3', 
                'Appliance4', 'Appliance5', 'Appliance6', 
                'Appliance7', 'Appliance8', 'Appliance9')
        for ind, col in enumerate(cols):
            a[ind] = self.data0[col]
            k[ind] = KMeans(n_clusters=2).fit(array(a[ind]).reshape(-1,1))
            print(f'{k[ind].cluster_centers_}')
            if diff(k[ind].cluster_centers_.reshape(1,-1))[0][0]>1e-5:
                isoff = a[ind][k[ind].labels_<1]
                ison = a[ind][k[ind].labels_>0]
            elif diff(k[ind].cluster_centers_.reshape(1,-1))[0][0]<-1e-5:
                isoff = a[ind][k[ind].labels_>0]
                ison = a[ind][k[ind].labels_<1]
            else:
                # cluster_centers_ may be [[0.], [0.]]
                with open(save_name, 'a') as f:
                    f.write('\t"{}":{'.format(col))
                    if col != 'Aggregate':
                        f.write('\n\t\t"name": "{}",'.format(self.app_name[ind-1]))
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
            with open(save_name, 'a') as f:
                f.write('\t"' + col + '":{')
                if col != 'Aggregate':
                    f.write('\n\t\t"name": "{}",'.format(self.app_name[ind-1]))
                f.write('\n\t\t"thrd": {}'.format(thrd))
                f.write(',\n\t\t"mean": {}'.format(mean))
                f.write(',\n\t\t"std": {}'.format(std))
                f.write(',\n\t},\n')
        with open(save_name, 'a') as f:
            f.write('},\n')

        return None


    def BM(self, 
           obj=0,
           sel_ac: str= None,
           no_plot: bool=False,
           no_show: bool=False,
           **args
           ):
        '''
        obj: str (app name) or int (1 ~ n)
            the Appliance on who the BM is caculated
            will be used as .loc[:, .columns!=obj]

            obj <- self.ins_name[obj] if is int
        
        sel_ac: str('active, reactive, apparent)
            select a specific ac_type to analysis
        no_plot: bool
            skip the plotting func
        no_show: bool
            plot (including save figs) but no `plt.show()`

        **args:
            for plotting func
        fig_types: Iterable = (), 
            figs to save in the types
        no_margin: bool=False
            draw heatmap only without margin marks if is True

        === good luck! ===
        '''

        print('\trun `BM()`!')
        # fix input
        if sel_ac is None:
            sel_ac = list(self.avail_ac)[0]
            print('select ac as `{}` by default'.format(sel_ac))

        if isinstance(obj, str):
            if not obj in self.ins_name[sel_ac]:
                print('`{}` is not acceptable!'.format(obj))
                obj = self.ins_name[sel_ac][0]
        elif isinstance(obj, int):
            obj -= 1
            if obj >= self.appQ[sel_ac]:
                obj = self.appQ[sel_ac]-1
                print('fix `obj` to {}'.format(obj))
            elif obj < 0:
                obj = 0
                print('fix `obj` to {}'.format(obj))
            obj = self.ins_name[sel_ac][obj]
            print('get caculated obj as `{}`'.format(obj))
        else:
            raise ValueError('get `obj` as {} is not acceptable'.format(obj))

        data0 = self.data0[sel_ac]
        data_s = data0[obj]
        # data remains
        data_r = data0.loc[:, data0.columns!=obj]

        thrd = 12
        data_ra = data_r[data_s>thrd]
        data_rb = data_r[data_s<=thrd]
        print('{} + {} = {}'.format(len(data_ra.index), len(data_rb.index), 
                                    len(data_ra.index)+len(data_rb.index)))

        '''
        extract data_ra and data_rb
        '''
        for key, datax in zip(('ra', 'rb'), [data_ra, data_rb]):
            key = '-'.join([obj, key])
            # self.data0[key] = data0
            # self.appQ[key] = len(data0.columns)
            # self.ins_name[key] = tuple(data0.columns)

            # self.data2[key] = gen_PKMap(self, key=key, model=self.model)
            if sel_ac == 'dd':
                self.data2[key] = gen_PKMap(self, data0=datax, model=self.model)
            else:
                self.data0[key] = datax
                self.data2[key] = gen_PKMap(self, data0=datax, key=key, model=self.model)
            self.avail_key.add(key)
            if False:
            # if not no_plot:
                self.plot(self, key=key, no_show=no_show, **args)
            # remove 0
            # self.data2[key] = {it[0]:(it[1] if it[1]>0 else 1) for it in self.data2[key].items()}

        data_ra = self.data2['-'.join([obj,'ra'])]
        data_rb = self.data2['-'.join([obj, 'rb'])]
        data2 = {k:data_ra[k] + data_rb[k] for k in data_ra.keys()}
        t_ra = sum(list(data_ra.values()))
        t_rb = sum(list(data_rb.values()))
        print('get (t_ra, t_rb) as ({}, {})'.format(t_ra, t_rb))
        self.t_ra = t_ra
        self.t_rb = t_rb
        sbm = {}
        for sc in data_ra.keys():
            # State-Combination, like '00110110' for appQ is 9
            if data2[sc] == 0:
                sbm[sc] = nan
            else:
                if data_ra[sc] == 0:
                    y1 = 0
                    y2 = log10(data_rb[sc])/log10(data2[sc])
                elif data_rb[sc] == 0:
                    y1 = log10(data_ra[sc])/log10(data2[sc])
                    y2 = 0
                else:
                    y1 = log10(data_ra[sc])/log10(data2[sc])
                    y2 = log10(data_rb[sc])/log10(data2[sc])
                    # y1 = data_ra[sc]/t_ra
                    # y2 = data_rb[sc]/t_rb
                # print('(y1, y2) is {}'.format((y1, y2)))
                # value of smb_ from -1 to +1
                sbm_ = (sqrt(2)*y1 + 1)/(sqrt(2)*y2 + 1) -sqrt(2)
                sbm[sc] = log(sbm_ + sqrt(2))/log(1+sqrt(2))
            # print('SBM[{}] <- {}'.format(sc, sbm_))
        bm = sum([abs(k) for k in sbm.values()])
        s = sum([k > 0 for k in data2.values()])
        bm /= s
        # s is the number of valid item in sbm map
        print('find {} valid items in SBM'.format(s))
        del s
        print('get BM as {}'.format(bm))

        key = '-'.join([obj, 'bm'])
        self.data2[key] = sbm
        self.bm[key] = bm
        self.avail_key.add(key)

        h1 = Hellinger(self.data2[key])
        print('=== D_pH  is {}'.format(h1))
        if not self.no_count:
            h2 = Hellinger(self.data2[key], self.data2['active'])
            print('=== D_pH2 is {}'.format(h2))
            self.Dph[key] = (h1, h2)
        else:
            self.Dph[key] = (h1, )

        # print('=== 2*D_pH is {}'.format(Hellinger(self.data2[key], data2)))

        if not no_plot:
            do_plot(self, key=key, no_show=no_show, **args)

        return bm

'''
def do1():
    
    with NoPrints():
        bm0, sbm0 = p1.BM(obj=2, sel_ac='active', no_plot=True)
    p1.appQ['dd'] = 9
    p1.app_name['dd'] = p1.app_name['active']
    p1.ins_name['dd'] = p1.ins_name['active']
    datas = []
    hs = []
    bms = []
    d0 = p1.data0['active']
    ys = set(d0.index.year)
    t0 = time()
    print(print('gogogo') and print(2-1))
    
    for y in ys:
        # the hash of int is neurally in order
        d0_y = d0.loc[d0.index.year==y]
        ms = set(d0_y.index.month)
        for m in ms:
            d0_m = d0_y.loc[d0_y.index.month==m]
            ds = set(d0_m.index.day)
            for d in ds:
                d0_x = d0_m.loc[d0_m.index.day==d]
                # set `end` to stay
                print('\r\t\treading {}'.format(d0_x.index[0].date()), end='')
                if d0_x.index[0].hour < 1 and d0_x.index[-1].hour > 22:
                    datas.append(d0_x)

                    with f():
                        p1.data0['dd'] = d0_x.copy()
                        p1.data2['dd'] = gen_PKMap(p1, key='dd')
                        bm, sbm = p1.BM(obj=2, sel_ac='dd', no_plot=True)
                        h = Hellinger(sbm0, sbm)
                    hs.append(h)
                    bms.append(bm)
                    
                else:
                    print('\t'*2+'find invalid time: {}, {:0>2}:{:0>2} to {:0>2}:{:0>2}'.format(
                        d0_x.index[0].date(), 
                        d0_x.index[0].hour, d0_x.index[0].minute, 
                        d0_x.index[-1].hour, d0_x.index[-1].minute
                        )+' '*22, end='')
    del d0_y, d0_m, d0_x
    print('\rcost {} '.format(beauty_time(time()-t0)+' '*64))
'''

if __name__ == "__main__":
    pass
