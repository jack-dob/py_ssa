# py_ssa #

Implementing 1D and 2D Singular Spectrum Analysis in python.


## Details ##

See [this paper](https://arxiv.org/pdf/1309.5050.pdf) for the algorithms I am trying to implement.


## Known Issues ##

* Something is a bit wrong with the 2D ssa implementation. I think I need to fiddle with the quasi_hankelisation function.


## TODO ##

* Fix 2D ssa implementation.
* Tidy up 2D ssa implementation, I should put the functions it currently uses as methods of SSA2D.
* Move the plotting routines to their own module to make it easier to ignore them when writing the algorithm.
* Test the FFT implementation of SVD vs the numpy implementation to see which is faster/takes more memory.
* Decide on how to implement grouping for 1D and 2D case. At the moment I either have elementary grouping or a couple of test cases just to make sure it works. Ideally I want to actually group the Trajectory matricies in a logical and useful way.


## Requirements ##

Numpy
: The heavy lifting is done with numpy arrays and routines.

Matplotlib
: For plotting/displaying tests, it could be ripped out easily.

Numba (OPTIONAL)
: Trying to get some speed-up by using Numba in various routines. I'm a newbie at using it, so I've made it an optional dependency. If you don't have it, the Numba decorators *should* change to an identity function.

