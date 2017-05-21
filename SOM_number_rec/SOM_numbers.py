import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import animation 
import pickle
import sys

try:
	epoc_times = int(sys.argv[1]) 
except IndexError:
	epoc_times = 10

Train_data = np.transpose(np.loadtxt("TrData.txt", delimiter=","))
Train_labels = np.loadtxt("TrLabel.txt", delimiter=",")

# 2d w
N = 10
M = 10

# 3d data
d = Train_data.shape[1]

np.random.seed(2017)


def main():
	# creates artists instances
	fig, ax = plt.subplots(figsize=[7,7])

	# generates new data
	def mesh_gen():
		# initializes the system
		w, sigma0, eta0, T_max, tau1, tau2 = initialize()
		for tt in range(epoc_times):
			print("epoch: {}".format(tt+1))
			for t in range(T_max):
				sigma , eta = time_evolution(t, sigma0, eta0, tau1, tau2)
				x = sampling(t)
				(i_win, j_win) = find_winner(w, x)
				yield (w, t)
				for i in range(N):
					for j in range(M):
						h = neighbor_func(i_win, j_win, i, j, sigma)
						alpha = eta*h
						w[i,j] = (1-alpha)*w[i,j,:] + alpha*x
			print("\n")

	for w, t in mesh_gen():
		flushPrint("current t: ", t+1)

	# print(w[0,0,0])
	labels = labeling(w)
	with open("SOM.pkl", "wb") as f1:
		pickle.dump(w, f1)
	with open("labels.pkl", "wb") as f2:
		pickle.dump(labels, f2)
	show_mesh(w, ax)
	plt.savefig("trained_map.png", dpi=300)
	plt.show()


def flushPrint(string,num):
	sys.stdout.write('\r')
	sys.stdout.write(string+'{}'.format(num))
	sys.stdout.flush()

def show_numbers(a, ax):
	# j: column number
	ax.imshow(a, 
		cmap="gray", interpolation="nearest")

def initialize():
	w = np.random.random([N,M,d])
	sigma0 = np.sqrt(M*M + N*N)/2
	eta0 = 0.1
	T_max = Train_data.shape[0]
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

def sampling(i):
	img_vec = Train_data[i,:]
	return img_vec

def labeling(w):
	labeled_mat = np.zeros([N,M])
	for i in range(N):
		for j in range(M):
			minimum = np.inf
			label = 0
			for ii in range(Train_data.shape[0]):
				dist = la.norm(w[i,j] - Train_data[ii])
				if dist <= minimum:
					minimum = dist
					label = Train_labels[ii]
			labeled_mat[i,j] = label
	return labeled_mat

def show_mesh(w, ax):

	U = np.transpose(w[0,0].reshape(28,28))
	for j in range(1, M):
		b = np.transpose(w[0,j].reshape(28,28))
		U = np.concatenate((U, b), axis = 1)

	for i in range(1, N):
		a = np.transpose(w[i,0].reshape(28,28))
		for j in range(1, M):
			b = np.transpose(w[i,j].reshape(28,28))
			a = np.concatenate((a, b), axis = 1)
		U = np.concatenate((U, a), axis = 0)

	for i in range(1, N):
		ax.plot([0, 280], [i*28, i*28], color="white")
	for j in range(1, M):
		ax.plot([j*28, j*28],[0, 280], color="white")

	ax.imshow(U, cmap="gray", interpolation="nearest")





	# for ii in range(N):
	# 	for jj in range(M):
	# 		for i in range(0, 28*N, 28):
	# 			for j in range(0, 28*M, 28):
	# 				mesh_mat[i:(i+28), j:(j+28)] \
	# 					= np.transpose(w[ii, jj].reshape(28,28))


if __name__=="__main__":
	# fig, ax = plt.subplots()
	# show_numbers(3, ax)
	# plt.show()
	# print(len(Train_data[0]))
	main()













