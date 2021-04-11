# the Pseudo Karnaugh Map

## Introduction

The Pseudo Karnaugh Map (PKMap),
a coloured Weighted / Multi-valued Karnaugh Map,
an Enchanced Karnaugh Map as the project's initially named,
is an evaluation method to visualize the imbalanced class distribution
of a dataset.

This project was started to guide the generating of synthesis datasets
in which a self-collected data is added into a public dataset.
Later, we found it works in dataset imbalance representation as well.

The codes is programmed and tested on Python 3.8
both on Windows 10 version 2004 and Mojave 10.14.6.
Since a new format feature introduced in [PEP498](https://www.python.org/dev/peps/pep-0498/)
is used, Python lower than 3.6 (inclusive) need a few extra adapting.

(The useage of this feature has been removed since update
update [bd9401a](https://github.com/PKMap/PKMap/commit/bd9401abe7d50ca91637d8a25bf43aaa0a1fce25))

## Why PKMap?

Gain instructive insight of a datset before utilizing
is currently virtually impossible.
The PKMap is expected to offer an intuitionistic and visualized recognization
of the imbalanced classes distribution of a dataset.

## Installation

The `pypi` support is coming.
A directly clone is always welcomed.

## How to use

First, you'll need to import a dataset (usually a .csv file)
using `read_REFIT()`.
We'll use REFIT (can be downloaded [here](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned))
as a demonstration from a relative path.

```python
from pkmap import PKMap

file_path = './REFIT/CLEAN_House17.csv'
obj = PKMap(file_path)
```

In update [bd9401a](https://github.com/PKMap/PKMap/commit/bd9401abe7d50ca91637d8a25bf43aaa0a1fce25),
we add support for `nilmtk.Buildings` item:

```python
from nilmtk import DataSet
from pkmap import PKMap

D1 = DataSet('refit.h5')
obj = PKMap(file=D1.Buildings[16])
```

(Ignore the part below if it doesn't make sense)

The `.PTb`, a pseudo truth table, is a dictionary of statistic result which looks like this:

```python
obj.PTb = {
    # ......
    '11100010': 53,
    '11100110': 627,
    # ......
},
```

The `.keys()` represents the ON/OFF state of each 9 appliances.
And the `.values()` means how many times the state combination
is counted.
Details about this has been explained in my latest paper
(reference will be updated soon)

A Karnaugh map can be displayed by:

```python
obj.plot()
```

![example of PKMap](figs/PKMap_REFIT_House17_active.svg)

BTW, the background with `patch=r'/'` is successfully added
in update [ddab932](https://github.com/PKMap/PKMap/commit/ddab9322d1139f090f1a1d4ccd5276a4acd15b58)
under the help of `Artist.set_zorder()`.

The colormap we used for default is `inferno_r`,
where the brigner means less, and the darker mean more.
This makes the 'lighter' parts looks like the background color.

In case you want to change the colormap, you can do:

```python
obj.plot(data, cmap='viridis_r')
```

Or, you can save the PKMap by offering a fig type
(str without dot or any Iterable item is supported):

```python
obj.plot(data, fig_types='png')
```

As the figure type will be passed to `matplotlib.pyplot.savefig`,
formats will not be supported except

```python
(eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)
```

Also, a default value of `("png", "svg")` will be used
if `fig_types` is set to `"d"` or `"default"`.

## the Balance Mertic Mapping

Experimental codes have been updated, and coming the paper.
the BM map looks like this by now:

![example of BM](figs/PKMap_REFIT_House17_Computer-bm.svg)

## previewing

A preview funtion on the whole 9 appilances of a single dataset
in REFIT has been accomplished bascilly.

==**This feature is currently unstable**==

Here is an example how to use it:

```python
p1 = PKMap('./REFIT/CLEAN_House16.csv', count=False)
p1.preview()
```

![an example of preview funcion](figs/preview_house16_by_day.svg)

The upper axis plots each appliance's data (including aggregate data)
in hours. The data are devided by mean of each applilance for
better color experience.
The lower axis plots the data slice in original data according to
where the user point out on the upper axis.
And the clicked part is hightlighted by a cyan rectangle for conspicuousness.

Besides, this preview tool can also help in evaluation assessment,
and represent where the moedel behavior better or worse and
the releative waves.

## Publications

A primiary paper about the PKMap has been publish, more is on the way.

1. Lu Z, Liu G, Liao R. A pseudo Karnaugh mapping approach
for datasets imbalance. In: E3S Web of Conferences. Vol 236. EDP Sciences;
2021:04006. DOI:[10.1051/e3sconf/202123604006](https://doi.org/10.1051/e3sconf/202123604006)

Here is the BibLaTeX code:

```text
@inproceedings{lu2021,
  title = {A Pseudo {{Karnaugh}} Mapping Approach for Datasets Imbalance},
  booktitle = {{{E3S Web}} of {{Conferences}}},
  author = {Lu, Zhijian and Liu, Gang and Liao, Rongwen},
  editor = {Anpo, M. and Song, F.},
  date = {2021},
  volume = {236},
  pages = {04006},
  publisher = {{EDP Sciences}},
  doi = {10.1051/e3sconf/202123604006},
  eventtitle = {3rd {{International Conference}} on {{Energy Resources}} and {{Sustainable Development}} ({{ICERSD2020}})}
}
```
