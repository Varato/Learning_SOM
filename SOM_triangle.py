import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import animation 
import sys

# 2d w
N = 5
M = 5

# 3d data
d = 2

np.random.seed(2017)

def main():
	# creates artists instances
	fig, ax = plt.subplots(figsize=[15,10])
	ax.set_xlim([-0.9*255,0.9*255])
	ax.set_ylim([-0.8*255,1*255])
	ax.plot(255*np.array([np.sqrt(2)/2,-np.sqrt(2)/2,0]),255*np.array([-np.sqrt(6)/6,-np.sqrt(6)/6,np.sqrt(6)/3]), "o", color="black", markersize=10)
	lines = [None]*(N+M)
	for i in range(N):
		lines[i], = ax.plot([], [], 'o-', color="cornflowerblue")
	for j in range(M):
		lines[N+j], = ax.plot([],[],'-', color="cornflowerblue")
	dot, = ax.plot([],[],"*", color="red", markersize=10)
	dot1, = ax.plot([],[],"o", color="red", markersize=10)
	text = ax.text(0.4, 1.02, "evolution times = {}".format(0))

	# init_func for animation
	def init():
		for line in lines:
			line.set_data([],[])
		dot.set_data([],[])
		dot1.set_data([],[])
		text.set_text("evolution times = {}".format(0))

		return (lines, dot, dot1, text)

	# static draw the mesh
	def draw_mesh(data):
		'''
		only for 2D SOM and for 2D data
		'''
		w, x0, y0 = data
		N, M, d = w.shape
		for i in range(N):
			xx = w[i,:,0]
			yy = w[i,:,1]
			lines[i].set_data(xx,yy)
		for j in range(M):
			xx = w[:,j,0]
			yy = w[:,j,1]
			lines[N+j].set_data(xx,yy)
		dot.set_data(x0[0], x0[1])
		dot1.set_data(y0[0], y0[1])

	# update function for animation
	def update(data):
		'''
		only for 2D SOM and for 2D data
		'''
		w, x0, y0, t = data
		N, M, d = w.shape
		for i in range(N):
			xx = w[i,:,0]
			yy = w[i,:,1]
			lines[i].set_data(xx,yy)
		for j in range(M):
			xx = w[:,j,0]
			yy = w[:,j,1]
			lines[N+j].set_data(xx,yy)
		dot.set_data(x0[0], x0[1])
		dot1.set_data(y0[0], y0[1])
		text.set_text("evolution times = {}".format(t))
		return tuple(lines)+(dot,dot1, t)

	# generates new data
	def mesh_gen():
		# initializes the system
		w, sigma0, eta0, T_max, tau1, tau2 = initialize()
		for t in range(T_max):
			flushPrint("current t: ", t)
			sigma , eta = time_evolution(t, sigma0, eta0, tau1, tau2)
			x = sampling()
			(i_win, j_win) = find_winner(w, x)
			y = w[i_win,j_win]
			yield (w, x, y, t)
			for i in range(N):
				for j in range(M):
					h = neighbor_func(i_win, j_win, i, j, sigma)
					alpha = eta*h
					w[i,j] = (1-alpha)*w[i,j,:] + alpha*x
		print("\n")

	anim=animation.FuncAnimation(fig, update, mesh_gen, init_func=init ,interval=100, repeat=True)
	# anim.save("SOM_square.mp4")
	plt.show()


def flushPrint(string,num):
	sys.stdout.write('\r')
	sys.stdout.write(string+'{}'.format(num))
	sys.stdout.flush()

def initialize():
	w = np.random.random([N,M,d])
	sigma0 = np.sqrt(M*M + N*N)/2
	eta0 = 0.1
	T_max = 500
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
	# samples from sides
	# r = np.random.random()
	x = 255*np.array([np.array([np.sqrt(2)/2,-np.sqrt(6)/6]), np.array([-np.sqrt(2)/2,-np.sqrt(6)/6]), np.array([0,np.sqrt(6)/3])])[np.random.choice(3)]
	return x
	# samples in the area

if __name__=="__main__":
	main()












