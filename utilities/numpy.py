#!/usr/bin/python3
"""
Contains utility functions to help with numpy routines
"""
import numpy as np
import typing

def slice_center(a, b):
	"""
	returns a slice of a in the shape of b about the center of a
	Accepts np.ndarray or tuple for each argument
	"""
	a_shape = np.array(a.shape if type(a) is np.ndarray else a)
	b_shape = np.array(b.shape if type(b) is np.ndarray else b)
	s_diff = a_shape - b_shape
	return(tuple([slice(d//2, s-d//2) for s, d in zip(a_shape,s_diff)]))

def slice_outer_mask(a, b):
	"""
	Returns a mask of array a that are not included an a
	region of shape b in the center
	"""
	a_shape = np.array(a.shape if type(a) is np.ndarray else a)
	b_shape = np.array(b.shape if type(b) is np.ndarray else b)
	x = np.ones(a_shape, dtype=bool)
	x[slice_center(a,b)] = False
	return(x)

def idx_center(a):
	"""
	Get center index of array "a", will return index before midpoint
	for even size axes
	"""
	return(tuple([x//2 + x%2 for x in a.shape]))

def idx_grid(a):
	return(np.mgrid[tuple([slice(0,s) for s in a.shape])])

def idxs_of_grid(a, as_tuples=True):
	idxs = np.reshape(idx_grid(a), (len(a.shape),a.size)).T
	if as_tuples:
		for idx in idxs:
			yield(tuple(idx))
	else:
		return(idxs)

def array_from_sequence(s, fill_value=0, dtype=None, order='C', like=None):
	"""
	Create an array from a sequence 's'.

	Will create a rectangular array of (hopefully) basic types, the 
	shape will be large enough to hold all the elements of "s", indices
	not covered by "s" will be set to 'fill_value'
	"""
	shape = get_iterable_shape(s)
	a = np.full(shape, fill_value=fill_value, dtype=dtype)
	fill_from(a, s)
	return(a)

def fill_from(a, s):
	"""
	Will fill the array 'a' with values from 's'. Does not change the values
	of 'a' where 'a' and 's' do not overlap.
	"""
	n = min(a.shape[0], len(s))
	if a.ndim == 1:
		a[:n] = s[:n]
		return(a)
	elif a.ndim > 1:
		for i in range(n):
			a[i,...] = fill_from(a[i],s[i])
	return


def get_iterable_shape(iterable):
	"""
	Returns the *rectangular* shape of 'iterable', this is the shape
	that is large enough to hold the maximum number of elements that
	iterable has in each dimension.
	"""
	s = []
	if hasattr(iterable, '__iter__'):
		s.append(len(iterable))
		el_s = []
		for el in iterable:
			if hasattr(el, '__iter__'):
				el_s += get_iterable_shape(el)
		if len(el_s) > 0:
			s.append(max(el_s))
	return(s)
