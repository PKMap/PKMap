# -*- coding: utf-8 -*-
# === pkmap.py ===

# from pandas.io.parsers import read_csv
from ekmapTK import do_plot, read_REFIT
from ekmapTK import do_plot

from re import findall

class pkmap(object):
    """
    docstring
    """
    
    def __init__(self, file_path, save_file=False, slice=None):
        """
        docstring
        """
        self.__file_path = file_path
        self.__save_file = save_file
        self.__slice = slice

        self.__house_name = findall('/(.+)\.', file_path)[0]
        self.__file_dir = '/'.join(file_path.split('/')[:-1])
        self.__house_number = findall('\d+', self.__house_name)[-1]

        self.load()


    def load(self):
        """
        docstring
        """
        self.data0 = read_REFIT(
            file_path = self.__file_path, 
            save_file = self.__save_file, 
            slice = self.__slice
        )

        return None


    def plot(self, **args):
        """
        docstring
        """
        do_plot(self.data0, **args)

        return None


