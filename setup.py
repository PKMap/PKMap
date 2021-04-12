
import codecs
import os

from setuptools import setup, find_packages

__version__ = '0.2.1'

THENAME = "pkmap"
DESCRIPTION = "a dataset visualization generator supposed to \
               exhibit the imbalance for human eye acquisition"
MAINTAINER = "ZJ Lewous"
MAINTAINER_EMAIL = "zj.lewous@gmail.com"
URL = "https://github.com/pkmap/pkmap"
LICENSE = "Mozilla"
DOWNLOAD_URL = "https://github.com/pkmap/pkmap/tarball/master"

setup(
    name = THENAME,
    version = __version__,
    packages = find_packages(),
    install_requires = [
        "numpy", 
        "pandas", 
        "matplotlib==3.1.3", 
        "seaborn", 
        "scikit-learn>=0.21.2", 
        "jupyter", 
        "tqdm", 
    ],
    extras_require = {
        "optional": [
            "nilmtk",
        ]
    },
    description = DESCRIPTION,
    author = MAINTAINER,
    author_email = MAINTAINER_EMAIL,
    url = URL,
    download_url = DOWNLOAD_URL,
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    # long_description = "see in " + URL, 
    license = LICENSE,
)
