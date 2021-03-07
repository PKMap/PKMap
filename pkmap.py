# -*- coding: utf-8 -*-
# === pkmap.py ===

import os
from re import findall, match
from copy import copy
from collections.abc import Iterable 

import matplotlib.pyplot as plt
from nilmtk import dataset
from numpy.lib.arraysetops import isin
from numpy.lib.npyio import save
from pandas import DataFrame 
from pandas import read_excel
from pandas import concat

from numpy import diff, array, log, log10
from numpy import sum, abs, sqrt
from sklearn.cluster import KMeans

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

from .ekmapTK import gen_PKMap2 as gen_PKMap
from .ekmapTK import read_REFIT2 as read_REFIT
from .ekmapTK import do_plot, do_plot_BM
from .house_preview import plot_time


class pkmap(object):
    """
    docstring
    """
    
    def __init__(self, file = None, 
                 model: str='thrd', 
                 save_file: bool=False, 
                 n_slice=None, 
                 no_count: bool=False,
                 no_load: bool=False,
                 ):
        """
        docstring

        file: .csv file path | nilmtk.Building
            input a data file and gogogo

        ===  do not use `no_load` for now ===
        """

        if file is None:
            # None(s) share same id
            print('file is None!')
            # here is a default test file
            self.file = './REFIT/CLEAN_House7.csv'
        else:
            self.file = file
        self.save_file = save_file
        self.model = model
        self.n_slice = n_slice
        self.no_count = no_count
        self.no_load = no_load
        self.cache_dir = os.path.join(os.getcwd(), 'cache')
        self.bm = {}
        # print(self.cache_dir)
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
            meters = self.file.elec.submeters().meters
            # self.appQ = len(meters)
            # self.app_name = tuple([m.appliances[0].identifier.type for m in meters])
            # instance name, used for pd.DataFrame
            # self.ins_name = tuple([m.appliances[0].label(pretty=True) for m in meters])
            self.load_nilm(model=self.model, n_slice=self.n_slice, no_count=self.no_count)
            self.isnilmtk = True
            
        else:

            print('====== get unknow type: {}! ======'.format(type(self.file)))

    def load_nilm(self, 
                  model: str='thrd', 
                  n_slice=None, 
                  no_count: bool=True):
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
        if 'power' in meters[0].available_physical_quantities():
            avail_ac = set()
            for meterx in meters:
                avail_ac |= {m.get('type') for m in meters[0].device['measurements']
                            if m['physical_quantity']=='power'}
            print('find available_ac: {}'.format(avail_ac))
            self.avail_ac = avail_ac
            # [print(m.available_ac_types('power')) for m in meters]

            if not self.no_load:
                main_ = next(self.file.elec.meters[0].load())
                st_day = main_.index[0].date()
                ed_day = main_.index[-1].date()
                m2 = None
                for ac1 in avail_ac:
                # for ac1 in ('reactive',):
                    m1 = [next(m.load(ac_type=ac1)).loc[st_day:ed_day] for m in meters 
                        if ac1 in m.available_ac_types('power')]
                    m2 = concat(m1, axis=1)
                    self.app_name[ac1] = [m.appliances[0].identifier.type for m in meters 
                                if ac1 in m.available_ac_types('power')]
                    self.ins_name[ac1] = tuple([m.appliances[0].label(pretty=True) for m in meters
                                        if ac1 in m.available_ac_types('power')])
                    m2.columns = self.ins_name[ac1]
                    self.data0[ac1] = m2
                    print(m2)
                    self.appQ[ac1] = len(m2.columns)
                    if not no_count:
                        self.data2[ac1] = gen_PKMap(self, key=ac1, model=model, n_slice=n_slice)
                self.len = len(m2.index)
            else:
                # `no_load` is True, repair `appQ`
                pass
        elif 'energy' in meters[0].available_physical_quantities():
            print('energy type found in {}'.format(self.file.metadata['dataset']))

        else:
            print('unknown physical_quantities: {}'.format(meters[0].available_physical_quantities()))

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
            titles="", pats=[]):
        """
        docstring

        !!!
            caution when offering `data2`
        !!!
        """

        # data_tp is always a dict
        if key.lower() in ('all', 'a', 'as'):
            print('plotting all ac_type: {}'.format(self.avail_ac))
            for key_ in self.avail_ac:
                # print('plotting: {}'.format(data2))
                do_plot(self, data2=None, key=key_, 
                        cmap=cmap, fig_types=fig_types, no_show=no_show,
                        titles=titles, pats=pats)
        else:
            
            # fix input
            if not key in self.avail_ac:
                print('key = `{}` is not available!'.format(key))
                key = list(self.avail_ac)[0]
            print('plotting ac_type as {}'.format(key))

            if data2 is None:
                if not self.data2:
                    if self.isnilmtk:
                        # self.load_nilm(model=self.model, n_slice=self.n_slice, no_count=False)
                        for ac1 in self.avail_ac:
                            self.data2[ac1] = gen_PKMap(self, key=ac1, model=self.model, n_slice=self.n_slice)
                    else:
                        self.load(no_count=False)
                data2 = self.data2[key]

            do_plot(self, data2=data2, key=key, 
                    cmap=cmap, fig_types=fig_types, 
                    no_show=no_show,
                    titles=titles, pats=pats)

        return None


    def preview(self, **args):
        """
        docstring
        """
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
           sel_ac = None,
           no_plot: bool=False,
           fig_types: Iterable=(),
           ):
        '''
        obj: str (app name) or int (1 ~ n)
            the Appliance on who the BM is caculated
            will be used as .loc[:, .columns!=obj]

            obj <- self.ins_name[obj] if is int
        
        sel_ac: str('active, reactive, apparent)
            select a specific ac_type to analysis

        === good luck! ===
        '''

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
        
        if not fig_types:
            fig_types = ('png', 'svg')


        data0 = self.data0[sel_ac]
        data_s = data0[obj]
        # data remains
        data_r = data0.loc[:, data0.columns!=obj]

        thrd = 12
        data_ra = data_r[data_s>thrd]
        data_rb = data_r[data_s<=thrd]
        print('{} + {} = {}'.format(len(data_ra.index), len(data_rb.index), 
                                    len(data_ra.index)+len(data_rb.index)))

        for key, data0 in zip(('ra', 'rb'), [data_ra, data_rb]):
            key = '('.join([obj, key])
            self.data0[key] = data0
            self.appQ[key] = len(data0.columns)
            self.ins_name[key] = tuple(data0.columns)
            self.avail_ac.add(key)

            self.data2[key] = gen_PKMap(self, key=key, model=self.model)
            if not no_plot:
                self.plot(key=key, fig_types=fig_types)
        
            # remove 0
            self.data2[key] = {it[0]:(it[1] if it[1]>0 else 1) for it in self.data2[key].items()}

        data_ra = self.data2['('.join([obj,'ra'])]
        data_rb = self.data2['('.join([obj, 'rb'])]
        t_ra = sum(list(data_ra.values()))
        t_rb = sum(list(data_rb.values()))
        print('get (t_ra, t_rb) as ({}, {})'.format(t_ra, t_rb))
        sbm = {}
        for sc in data_ra.keys():
            # State-Combination, like '00110110' for appQ is 9
            y1 = log10(data_ra[sc])/log10(t_ra)
            y2 = log10(data_rb[sc])/log10(t_rb)
            # y1 = data_ra[sc]/t_ra
            # y2 = data_rb[sc]/t_rb
            # print('(y1, y2) is {}'.format((y1, y2)))
            sbm_ = (sqrt(2)*y1 + 1)/(sqrt(2)*y2 + 1) -sqrt(2)
            sbm[sc] = log(sbm_ + sqrt(2))/log(1+sqrt(2))
            # print('SBM[{}] <- {}'.format(sc, sbm_))
        bm = sum(abs(list(sbm.values())))
        bm /= 2**(self.appQ[sel_ac]-1)
        print('get BM as {}'.format(bm))

        key = '('.join([obj, 'bm'])
        self.data2[key] = sbm
        self.bm[key] = sbm
        do_plot_BM(self, key=key, fig_types=('.png', '.svg'), no_show=no_plot)

        return bm


if __name__ == "__main__":
    pass
