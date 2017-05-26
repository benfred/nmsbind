### This project has been integrated into NMSLIB!!

```pip install nmslib``` should install the latest version of these bindings from the main [NMSLIB repo](https://github.com/searchivarius/nmslib) - please use that instead.

nmsbind
=======


[![Build Status](https://travis-ci.org/benfred/nmsbind.svg?branch=master)](https://travis-ci.org/benfred/nmsbind)
[![Windows Build
status](https://ci.appveyor.com/api/projects/status/025rl7knj2m62hs5?svg=true)](https://ci.appveyor.com/project/benfred/nmsbind)

nmsbind: Alternate Python Bindings for NMSLIB

This project uses [pybind11](https://github.com/pybind/pybind11) to
build Python bindings for [the Non-Metric Space Library (NMSLIB)](https://github.com/searchivarius/nmslib).

NMSLIB is a great library that provides many different methods for calculating approximate nearest neighbours.
Some of these methods are up to [10 times faster](https://raw.githubusercontent.com/searchivarius/nmslib/master/docs/figures/glove.png)
than provided by libraries like [Annoy](https://github.com/spotify/annoy). However, Annoy is currently much
more popular - at least in part because its easier to install and use from Python. 

This project aims to fix some of the hassles of using NMSLIB in Python, by using pybind to write alternate
bindings, and to work on making the install work seamlessly across multiple different environments.

Some advantages of this approach are:
 * Works with Python 3.5+ and Python 2.7+
 * Works on Linux / OSX / Windows systems
 * Easily installable with pip: 'pip install nmsbind' will download from pypi and install
 * More natural Python API:
    * the index is a class with methods (instead of getting a memory location and using global functions to access)
    * methods have sensible default parameters
    * docstrings provide some basic documentation on how to call
    * no need to manually free memory with ```nmslib.freeIndex```

To install:

```
pip install nmsbind
```

Basic usage:

```python
import nmsbind
import numpy

# create a random matrix to index
data = numpy.random.randn(10000, 100).astype(numpy.float32)

# initialize a new index, using a HNSW index on Cosine Similarity
index = nmsbind.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex({'post': 2}, print_progress=True)

# query for the nearest neighbours of the first datapoint
ids, distances = index.knnQuery(data[0], k=10)

# get all nearest neighbours for all the datapoint
# using a pool of 4 threads to compute
neighbours = index.knnQueryBatch(data, k=10, num_threads=4)
```

This library has been tested with Python 2.7 and 3.5/3.6. Running 'tox' will
build and run unittests on all these versions.
