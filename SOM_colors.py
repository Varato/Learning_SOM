import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import animation 
import sys

# 2d w
N = 50
M = 50

# 3d data
d = 3

np.random.seed(2017)

def main(data):
	# creates artists instances
	fig, (ax, ax1) = plt.subplots(ncols=2,figsize=[15,10])
	ax.set_xticks([])
	ax.set_xticks([])
	ax.set_title("SOM_colors")
	img = ax.imshow(np.array([[[0,0,0]]]), interpolation = "nearest")

	# init_func for animation
	def init():
		img.set_data(np.array([[[0,0,0]]]))
		return img
	# update function for animation
	def update(data):
		'''
		only for 2D SOM and for 2D data
		'''
		img.set_data(data)
		return img

	# generates new data
	def mesh_gen():
		# initializes the system
		w, sigma0, eta0, T_max, tau1, tau2 = initialize()
		for t in range(T_max):
			flushPrint("current t: ", t)
			sigma , eta = time_evolution(t, sigma0, eta0, tau1, tau2)
			x = sampling(data)
			(i_win, j_win) = find_winner(w, x)
			# yield w
			for i in range(N):
				for j in range(M):
					h = neighbor_func(i_win, j_win, i, j, sigma)
					alpha = eta*h
					w[i,j] = (1-alpha)*w[i,j,:] + alpha*x
		return w
		print("\n")

	# anim=animation.FuncAnimation(fig, update, mesh_gen, init_func=init ,interval=100, repeat="False")
	w = mesh_gen()

	for i in range(N):
		for j in range(M):
			d_red = la.norm(w[i,j] - data[0])
			d_green = la.norm(w[i,j] - data[1])
			d_blue = la.norm(w[i,j] - data[2])
			ddd = [d_red, d_green, d_blue]
			w[i,j] = data[ddd.index(min(ddd))]



	U_mat = np.zeros(w.shape)
	for i in range(N):
		for j in range(M):
			d_l = -1 if j==0 else la.norm(w[i,j]-w[i,j-1])
			d_r = -1 if j==M-1 else la.norm(w[i,j]-w[i,j+1])
			d_u = -1 if i==0 else la.norm(w[i,j]-w[i-1,j])
			d_d = -1 if i==N-1 else la.norm(w[i,j]-w[i+1,j])
			dists = [dd for dd in (d_l, d_r, d_u, d_d) if dd != -1]
			U_mat[i,j] = np.mean(dists)
	print(U_mat)
	ax.imshow(w, interpolation='nearest')
	ax1.imshow(1-U_mat,cmap="gray",interpolation='nearest')

	plt.show()

def initialize():
	w = np.random.random([N,M,d])
	sigma0 = np.sqrt(M*M + N*N)/2
	eta0 = 0.1
	T_max = 300
	tau1 = T_max/np.log(sigma0)
	tau2 = T_max
	return (w, sigma0, eta0, T_max, tau1, tau2)

def neighbor_func(i_win, j_win, i, j, sigma):
	d = la.norm([i_win-i, j_win-j])
	h = np.exp( -0.5*d*d/(sigma**2) )
	return h

def time_evolution(t, sigma0, eta0, tau1, tau2):
	'''
	update effective width and learning rate
	'''
	return (sigma0*np.exp(-t/tau1), 
		eta0*np.exp(-t/tau2))

def find_winner(w, in_put):
	min_dist = np.inf
	i_win = 0
	j_win = 0
	for i in range(N):
		for j in range(M):
			dist = la.norm(w[i,j]-in_put) 
			if dist <= min_dist:
				min_dist = dist
				i_win = i
				j_win = j
	return (i_win, j_win)

def sampling(data=None):
	'''
	data: row-wisely stored
	'''
	return data[np.random.choice(len(data))]

def flushPrint(string,num):
	sys.stdout.write('\r')
	sys.stdout.write(string+'{}'.format(num))
	sys.stdout.flush()

if __name__=="__main__":
	data = np.array([[1,0,0],[0,1,0],[0,0,1]])
	main(data=data)












