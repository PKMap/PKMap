# -*- coding: utf-8 -*-
# === pkmap.py ===

# from pandas.io.parsers import read_csv
from ekmapTK import do_plot, read_REFIT
from ekmapTK import do_plot
from ekmapTK import do_plot_time

from re import findall, match

import matplotlib.pyplot as plt
# from pandas import DataFrame 
from pandas import read_excel

import os
from copy import copy


class pkmap(object):
    """
    docstring
    """
    
    def __init__(self, file_path = None, save_file=False, slice=None):
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
        self.house_number = findall(r'\d+', self.house_name)[-1]
        self.load()
        self.app_name = read_excel(os.path.join(self.file_dir, 'MetaData_Tables.xlsx'), 
                    sheet_name='House ' + str(self.house_number), 
                    usecols=('Aggregate', ), ).values[:]
        # cleaning as having '???' inside
        self.app_name = tuple(['Unknown' if match(r'\?+', n) else n.replace(' ', '_')
                        for n in self.app_name.reshape(1,-1)[0]])
        # using as name_app[n][0]
        self.appQ = len(self.app_name)


    def load(self):
        """
        docstring
        """
        # data2: pseudo truth table
        # data0: original csv data
        self.data2, self.data0 = read_REFIT(
            file_path = self.file_path, 
            save_file = self.save_file, 
            slice = self.slice
        )

        return None


    def plot(self, **args):
        """
        docstring
        """
        do_plot(self.data2, **args)

        return None


    def plot_time(self, **args):
        """
        docstring
        """
        do_plot_time(self, **args)
        # pass

        return None


