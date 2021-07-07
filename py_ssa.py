#!/usr/bin/python3
"""
Implementation of Singular Spectrum Analysis.

See https://arxiv.org/pdf/1309.5050.pdf for details.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utilities as ut
import utilities.numpy 
import utilities.plt
import py_svd as py_svd
import typing

try:
	import numba
except ImportError:
	import types
	print('WARNING: Could not import package "numba". This disable JIT compilation optimisations.')
	def identity_decorator(x, *args, **kwargs):
		#print('DEBUGGING: In "identity_decorator()"')
		idf = lambda x: x
		if type(x) is type(idf):
			return(x)
		return(lambda x: x)
	numba = types.SimpleNamespace()
	numba.njit = identity_decorator # pass through function without doing anything


class SSA:
	def __init__(self, a, w_shape=None, svd_strategy='numpy', rev_mapping='fft'):
		self.a = a
		self.nx = self.a.size
		self.lx = self.nx//4 if w_shape is None else w_shape
		self.kx = self.nx - self.lx + 1
		
		
		self.X = self.embed(self.a)
		
		if svd_strategy == 'numpy':
			# numpy way
			# get svd of trajectory matrix
			self.u, self.s, self.v_star = np.linalg.svd(self.X)
			# make sure we have the full eigenvalue matrix, not just the diagonal
			self.s = py_svd.rect_diag(self.s, (self.lx,self.kx))
		
		if svd_strategy == 'eigval':
			S = self.X @ self.X.T
			s, self.u = np.linalg.eig(S)
			v = (self.X.T @ self.u)/np.sqrt(s)
			self.v_star = np.zeros((self.kx,self.kx))
			self.v_star[:self.lx,:self.kx] = v.T
			self.s = np.zeros((self.lx, self.kx))
			self.s[:self.u.shape[0], :self.u.shape[1]] = np.sqrt(np.diag(s))

		
		self.X_decomp = py_svd.decompose_to_matricies(self.u,self.s,self.v_star)
		self.d = self.X_decomp.shape[0]
		
		self.grouping =  self.get_grouping()
		
		self.group()
		
		if rev_mapping=='fft':
			self.reverse_mapping_fft()
		else:
			self.reverse_mapping()
		
		self.X_reconstructed = np.sum(self.X_ssa, axis=0)
		
		return

	def embed(self, a):
		X = np.zeros((self.lx, self.kx))
		for k in range(self.kx):
				X[:, k] = a[k:k+self.lx]
		return(X)
	
	def get_grouping(self):
		# no-op for now
		grouping = [[_x] for _x in range(0,self.d,1)]
		return(grouping)

		# test grouping op
		grouping = [[_x,_x+1] for _x in range(0,self.d,2)]
		return(grouping)
		
		# test better grouping op
		z = 1-self.d%2
		grouping = [
			np.array([0]), 
			*list(zip(*np.stack([np.arange(1,self.d-z,2),np.arange(2,self.d,2)]))),
			np.arange(self.d-z, self.d)
		]
		return(grouping)
	
	def group(self):
		# set up constants
		self.m = len(self.grouping)
		#self.m, self.I_max = self.grouping.shape
		# don't group if we don't have to
		s_dash = np.zeros((self.kx,self.kx))
		s_dash[:self.lx,:self.kx] = self.s
		
		
		if np.all(self.grouping == np.arange(0,self.X_decomp.shape[0])[:,None]):
			print('DEBUGGING: no op grouping')
			self.X_g = self.X_decomp
			self.u_g = self.u
			self.v_star_g = self.v_star
			self.s_g = self.s
			return
		
		
		print('DEBUGGING: pay attention to groups')
		self.X_g = np.zeros((self.m, self.lx, self.kx))
		self.u_g = np.zeros((self.lx, self.m))
		self.v_star_g = np.zeros((self.m, self.kx))
		self.s_g = np.zeros((self.m,self.m))
		for _m, I in enumerate(self.grouping):
			for j in I:
				self.X_g[_m] += self.X_decomp[j]
				self.u_g[:,_m] += self.u[:,j]
				self.v_star_g[_m,:] += self.v_star[j,:]
				self.s_g[_m,_m] += self.s[j,j]
		return
	
	
	def reverse_mapping_fft(self):
		print('DEBUGGING: IN "reverse_mapping_fft()"')
		self.X_ssa = np.zeros((self.m, self.nx))
		#self.X_dash = np.zeros_like(self.X_ssa)
		
		
		# extend u and v_star to size nx with zeros
		u_dash = np.zeros((self.nx,))
		v_dash = np.zeros((self.nx,))
		
		self.u_dash = np.zeros((self.m,self.nx,))
		self.v_dash = np.zeros((self.m,self.nx,))
		
		l_star = min(self.lx, self.kx)
		W = np.convolve(np.ones((self.kx)), np.ones((self.lx,)), mode='full')
		"""
		W = np.zeros((self.nx,))
		W[l_star-1:-(l_star-1)] = l_star
		W[:l_star-1] = np.arange(1,l_star)
		W[-(l_star-1):] = np.arange(1,l_star)[::-1]
		"""
		#self.W = W
		
		for g in range(self.m):
			u_dash[:self.lx] = self.u_g[:,g]
			v_dash[:self.kx] = self.v_star_g[g,:]
			s_dash = self.s[g,g]
			self.u_dash[g] = u_dash
			self.v_dash[g] = v_dash
			u_dash_fft = np.fft.fft(u_dash)
			v_dash_fft = np.fft.fft(v_dash)
			X_dash = np.fft.ifft(u_dash_fft*v_dash_fft*s_dash)
			#self.X_dash[g] = X_dash
			self.X_ssa[g] = (X_dash/W)#*(self.s_g[g,g])#*(self.nx/self.lx)
		return
	
	def reverse_mapping(self):
		print('DEBUGGING: IN "reverse_mapping()"')
		E = np.zeros((self.nx,))
		T_E = np.zeros((self.lx, self.kx))
		self.X_ssa = np.zeros((self.m, self.nx))
		
		#self.f_ip_vec = np.zeros_like(self.X_ssa)
		#self.f_norm_sq_vec = np.zeros_like(self.X_ssa)
		#self.T_E_vec = np.zeros((self.m, self.nx, self.lx, self.kx))
		for g in range(self.m):
			print(g)
			for n in range(self.nx):
				#print('\t',k)
				E[n] = 1
				T_E[...] = self.embed(E)
				self.T_E_vec[g,n,...] = T_E
				f_ip = frobenius_inner_prod(self.X_g[g], T_E)
				f_norm_sq = frobenius_norm(T_E)**2
				#self.f_ip_vec[g,n] = f_ip
				#self.f_norm_sq_vec[g,n] = f_norm_sq
				#self.X_ssa[g,n] = f_ip/f_norm_sq
				E[n] = 0
		return
	
	
	def plot_svd(self, recomp_n=None):
		if recomp_n is None:
			recomp_n = self.X_decomp.shape[0]
		py_svd.plot(self.u, self.s, self.v_star, self.X, self.X_decomp, recomp_n=recomp_n)
		return
	
	
	def plot_ssa(self, n_max=5):
		# Matplotlib parameters
		mpl.rcParams['lines.linewidth'] = 1
		mpl.rcParams['font.size'] = 8
		mpl.rcParams['lines.markersize'] = 2

		n = min(self.m, n_max if n_max is not None else self.m)
		f1, a1 = ut.plt.figure_n_subplots(n+3)
		a1=a1.flatten()
		f1.suptitle(f'First {n} ssa components of data (of {self.m})')
		ax_iter=iter(a1)
		
		
		ax = next(ax_iter)
		ax.set_title('Original and Reconstruction')
		ax.plot(self.a, label='data',ls='-',marker='.')
		ax.plot(self.X_reconstructed, label='sum(X_ssa)', ls='--', alpha=0.8)
		ax.legend()
		
		ax = next(ax_iter)
		ax.set_title('Residual')
		ax.plot(self.a - self.X_reconstructed, ls='-', marker='')
		
		ax = next(ax_iter)
		ax.set_title('Ratio (original/reconstruction)')
		ax.plot(self.a/self.X_reconstructed, ls='-', marker='')
		
		for i in range(n):
			ax=next(ax_iter)
			ax.set_title(f'X_ssa[{i}]\neigenvalue {self.s_g[i,i]:07.2E}')
			#ut.plt.remove_axes_ticks_and_labels(ax)
			ax.plot(self.X_ssa[i])
		return
			



#%%
class SSA2D:
	__slots__ = (
		'a',
		'lx',
		'ly',
		'nx',
		'ny',
		'kx',
		'ky',
		'lxly',
		'kxky',
		'u',
		's',
		'v_star',
		'X',
		'X_decomp',
		'grouping',
		'X_g',
		'X_ssa'
	)
	
	def __init__(self, a, w_shape=None):
		"""
		Set up initial values of useful constants
		
		a [nx,ny]
			Array to operate on, should be a type that numba recognises. If in
			doubt, use np.float64.
		w_shape [2]
			<int,int> Window shape to use for SSA, no array is actually created
			from this shape, it's used as indices to loops etc. If not given,
			will use a.shape//4 as window size
		"""
		self.a = a
		self.nx, self.ny = a.shape
		if w_shape is None:
			self.lx, self.ly = self.nx//4, self.ny//4
		else:
			self.lx, self.ly = w_shape
			
		self.kx, self.ky = self.nx-self.lx+1, self.ny-self.ly+1
		self.lxly = self.lx*self.ly
		self.kxky = self.kx*self.ky
		return
	
	def __call__(self):
		"""
		Perform whole ssa chain and return result
		"""
		
		# embed input matrix into trajectory matrix
		self.X = embed(self.a, (self.lx, self.ly))
	
		"""
		# numpy way
		# get svd of trajectory matrix
		self.u, self.s, self.v_star = np.linalg.svd(self.X)
		# make sure we have the full eigenvalue matrix, not just the diagonal
		self.s = svd.rect_diag(self.s, (self.lxly,self.kxky))
		"""
		
		# eigenvalues first way
		s, self.u = np.linalg.eig(self.X@self.X.T)
		v = np.zeros((self.kxky, self.kxky))
		v[:self.kxky, :self.lxly] = (self.X.T@self.u / np.sqrt(s))
		self.v_star = v.T
		self.s = np.zeros((self.lxly, self.kxky))
		self.s[:self.u.shape[0], :self.u.shape[1]] = np.diag(s) # maybe square root this?
		
		print(self.u.shape)
		print(self.s.shape)
		print(self.v_star.shape)
		
		
		
		# get the decomposed trajectories
		self.X_decomp = py_svd.decompose_to_matricies(self.u,self.s,self.v_star)
		
		# determine optimal grouping
		self.grouping = self.get_grouping()
		
		# group trajectory components
		self.X_g = group(self.X_decomp, self.grouping)
		
		
		#return(diagsums(self.u, self.v_star, self.lx, self.kx, self.nx, self.ny))
		
		# need to multiply by self.s to get correct normalisation
		self.X_ssa = quasi_hankelisation(self.u*np.diag(self.s), self.v_star, self.lx, self.ly, self.kx, self.ky, self.nx, self.ny)
		return(self.X_ssa)
		
		"""
		# slow, try not to use
		# reverse the mapping to get ssa of input matrix
		self.X_ssa = reverse_mapping(self.X_g, (self.nx,self.ny), (self.lx,self.ly))
		"""
		
		# return ssa
		return(self.X_ssa)
		
		
	def get_grouping(self):
		"""
		Define how the eigentriples (evals, evecs, fvecs) should be grouped.
		"""
		# LAZY IMPLEMENTATION
		# don't bother with determining grouping right now.
		return(np.zeros((0,0),dtype=int))
		
	
	def plot_all(self, n_max=36):
		self.plot_svd()
		self.plot_eigenvectors(n_max)
		self.plot_factorvectors(n_max)
		self.plot_trajectory_decomp(n_max)
		self.plot_trajectory_groups(n_max)
		self.plot_ssa(n_max)
		return

	def plot_svd(self, recomp_n=None):
		if recomp_n is None:
			recomp_n = self.X_decomp.shape[0]
		py_svd.plot(self.u, self.s, self.v_star, self.X, self.X_decomp, recomp_n=recomp_n)
		return
	
	def plot_eigenvectors(self, n_max=None):
		flip_ravel = lambda x: np.reshape(x.ravel(order='F'), x.shape)
		# plot eigenvectors and factor vectors
		n = min(self.u.shape[0], n_max if n_max is not None else self.u.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} Eigenvectors of X (of {self.u.shape[0]})')
		ax_iter=iter(a1)
		for i in range(n):
			ax=next(ax_iter)
			ax.set_title(f'i = {i} eigenval = {self.s[i,i]:07.2E}')
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(np.reshape(self.u[i,:],(self.lx,self.ly)).T)
		return
	
	def plot_factorvectors(self, n_max=None):
		flip_ravel = lambda x: np.reshape(x.ravel(order='F'), x.shape)
		n = min(self.v_star.shape[0], n_max if n_max is not None else self.v_star.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} Factorvectors of X (of {self.v_star.shape[0]})')
		ax_iter = iter(a1)
		for j in range(n):
			ax=next(ax_iter)
			ax.set_title(f'j = {j}')
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(flip_ravel(np.reshape(self.v_star[j,:],(self.kx,self.ky)).T))
			
	def plot_trajectory_decomp(self, n_max=None):	
		# Plot components of image decomposition
		n = min(self.X_decomp.shape[0], n_max if n_max is not None else self.X_decomp.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1 = a1.ravel()
		f1.suptitle('Trajectory matrix components X_i [M = sum(X_i)]')
		ax_iter = iter(a1)
		for i in range(n):
			ax = next(ax_iter)
			ax.set_title(f'i = {i}', y=0.9)
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_decomp[i], origin='lower', aspect='auto')
		return
		
		
		
	def plot_trajectory_groups(self, n_max=None):
		# plot elements of X_g
		n = min(self.X_g.shape[0], n_max if n_max is not None else self.X_g.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1 = a1.ravel()
		f1.suptitle('Trajectory matrix groups X_g [X = sum(X_g_i)]')
		ax_iter=iter(a1)
		for i in range(n):
			ax = next(ax_iter)
			ax.set_title(f'i = {i}', y=0.9)
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_g[i], origin='lower', aspect='auto')
		return
		
	def plot_ssa(self, n_max=None):
		# plot SSA of image
		n = min(self.X_ssa.shape[0], n_max if n_max is not None else self.X_ssa.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} ssa images of obs (of {self.X_ssa.shape[0]})')
		ax_iter=iter(a1)
		for i in range(n):
			ax=next(ax_iter)
			ax.set_title(f'i = {i}')
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_ssa[i])
		return
	
	
	
@numba.njit("f8[:](f8[:,:])")
def vectorise_mat(a : np.ndarray) -> np.ndarray:
	# numba doesn't like flatten(order='F'), so transpose first
	return(a.T.ravel())
	#return(a.ravel())


def unvectorise_mat(a, m):
	"""
	* faster without JIT
	"""
	#return(np.reshape(a.T, (m,a.size//m)))
	return(np.reshape(a, (m,a.size//m)).T)

@numba.njit("f8[:,:](f8[:,:], UniTuple(u8, 2))")
def embed(a, w_shape):
	lx, ly = w_shape
	nx, ny = a.shape
	kx, ky = nx-lx+1, ny-ly+1
	X = np.zeros((lx*ly, kx*ky)) 
	for k in range(kx):
		for l in range(ky):
			X[:,k+l*kx] = vectorise_mat(a[k:k+lx,l:l+ly])
	return(X)

def group(X_decomp, idx_array=np.zeros((0,0),dtype=int)):
	"""
	* faster without JIT
	
	X_decomp is the d images you get from SVD of X, where d is the number of singular values
	idx_array is a 2d array that associates each X_decomp_j with a group of such images
	
	idx_array.shape=(m,I_max)
		m - number of grouped images to get out
		I_max - maximum number of X_decomp images in each group
	The elements of idx_array are the indicies of the X_decomp images to put
	into a group, if an idx is -ve it is ignored
	"""
	d, lxly, kxky = X_decomp.shape
	m, I_max = idx_array.shape
	
	# don't bother grouping if we have nothing to group
	if m==0 or I_max==0 or d==m:
		return(X_decomp)
	
	X_g = np.zeros((m, lxly, kxky))
	for I in range(m):
		for j in range(I_max):
			idx = idx_array[I,j]
			if idx < 0:
				continue
			X_g[I] += X_decomp[j]
	return(X_g)

def frobenius_norm(A):
	"""
	Root of the sum of the elementwise squares
	"""
	return(np.sqrt(np.sum(A*A)))

def frobenius_inner_prod(A,B):
	"""
	Sum of the elementwise multiplication
	"""
	return(np.sum(A*B))

def reverse_mapping(X_g, original_shape, window_shape) -> np.ndarray :
	"""
	* faster without JIT
	
	Reverses the embedding map
	shape should be (nx,ny)
	"""
	nx, ny = original_shape
	lx, ly = window_shape
	kx, ky = nx-lx+1, ny-ly+1
	
	E = np.zeros((nx,ny))
	T_E = np.zeros((lx*ly, kx*ky))
	X_ssa = np.zeros((X_g.shape[0],nx,ny))
	
	for g in range(X_g.shape[0]):
		for k in range(nx):
			for l in range(ny):
				E[k,l] = 1
				T_E[...] = embed(E, window_shape)
				X_ssa[g,k,l] = frobenius_inner_prod(X_g[g], T_E)/(frobenius_norm(T_E)**2)
				E[k,l] = 0
	return(X_ssa)


def diagsums(u_j, v_j, lx, ky, nx, ny):
	u_dash = unvectorise_mat(u_j, lx)
	v_dash = unvectorise_mat(v_j, ky)
	u_dash_fft = np.fft.fft2(u_dash, (nx,ny))
	v_dash_fft = np.fft.fft2(v_dash, (nx,ny))
	ds = np.fft.ifft2(u_dash_fft*v_dash_fft, (nx,ny)).real
	return(ds.real)
	

def quasi_hankelisation(u, v_star, lx, ly, kx, ky, nx, ny):
	v_star = v_star
	X_ssa = np.zeros((lx*ly, nx, ny))
	W = np.zeros((nx,ny))
	X_dash = np.zeros((nx,ny))
	_ol = np.ones((lx*ly,))
	_ok = np.ones((kx*ky,))
	for j in range(lx*ly):
		W = diagsums(_ol, _ok, lx, ky, nx, ny)
		X_dash = diagsums(u[:,j], v_star[:,j], lx, ky, nx, ny)#maybe times one of u or v by s?
		X_ssa[j] = X_dash/W
	return(X_ssa)

def mult_by_quasi_hankel(A, b, bx, cxcy):
	nx, ny = A.shape
	b_dash = unvectorise(b, bx)
	A_fft = np.fft.fft2(A, (nx,ny))
	b_dash_fft = np.fft.fft2(b_dash, (nx,ny))
	c_dash = np.fft.ifft2(A_fft*np.conjugate(b_dash_fft), (nx,ny)).T
	return
	
#%%
if __name__=='__main__':
	# get example data
	try:
		import PIL
		print('Creating mandelbrot fractal as test case')
		obs = np.asarray(PIL.Image.effect_mandelbrot((60,50),(0,0,1,1),100))
	except ImportError:
		print('Creating random numbers as test case')
		obs = np.random.random((60,50))
	#obs, psf = fitscube.deconvolve.helpers.get_test_data()
	
	ssa2d = SSA2D(obs.astype(np.float64), (5,5))
	
	obs_ssa = ssa2d()
	
	# plot SSA of input data
	
	#ssa2d.plot_all()
	#ssa2d.plot_eigenvectors()
	#ssa2d.plot_factorvectors(25)
	ssa2d.plot_ssa()
	
	plt.show()
	
	# TESTING
	np.random.seed(100)
	n = 1000
	w = 10
	data = np.convolve(np.random.random((n,)), np.ones((w,)), mode='same')
	mean = np.mean(data)
	
	ssa = SSA(data, svd_strategy='numpy', rev_mapping='fft')
	#ssa.plot_svd()
	ssa.plot_ssa()
	
