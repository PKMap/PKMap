# -*- coding: utf-8 -*-
# === pkmap.py ===

import os
from re import findall, match
from copy import copy

import matplotlib.pyplot as plt
from numpy.lib.npyio import save
# from pandas import DataFrame 
from pandas import read_excel

from numpy import diff, array
from sklearn.cluster import KMeans


from ekmapTK import do_plot, read_REFIT
from ekmapTK import do_plot
from ekmapTK import do_plot_time
from house_preview import plot_time


class pkmap(object):
    """
    docstring
    """
    
    def __init__(self, file_path = None, save_file=False, slice=None, 
                count=True):
        """
        docstring
        """

        if file_path is None:
            # None(s) share same id
            print('file_path is None!')
            self.file_path = './REFIT/CLEAN_House7.csv'
        else:
            self.file_path = file_path
        self.save_file = save_file
        self.slice = slice
        self.file_dir = '/'.join(self.file_path.split('/')[:-1])
        self.house_name = findall(r'House\d+', self.file_path)[-1]
        self.house_number = int(findall(r'\d+', self.house_name)[-1])
        self.load(count)
        self.app_name = read_excel(os.path.join(self.file_dir, 'MetaData_Tables.xlsx'), 
                    sheet_name='House ' + str(self.house_number), 
                    usecols=('Aggregate', ), ).values[:]
        # cleaning as having '???' inside
        self.app_name = tuple(['Unknown' if match(r'\?+', n) else n.replace(' ', '_')
                        for n in self.app_name.reshape(1,-1)[0]])
        # using as name_app[n][0]
        self.appQ = len(self.app_name)
        self.len = len(self.data0.index)


    def load(self, count):
        """
        docstring
        """
        # data2: pseudo truth table
        # data0: original csv data
        self.data2, self.data0 = read_REFIT(
            file_path = self.file_path, 
            save_file = self.save_file, 
            slice = self.slice,
            count=count,
        )

        return None


    def plot(self, **args):
        """
        docstring
        """
        do_plot(self.data2, **args)

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


if __name__ == "__main__":
    pass
