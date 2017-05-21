import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import animation 
import pickle
import sys

N=10
M=10
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

def load_data():
	with open("SOM.pkl", "rb") as f1:
		w = pickle.load(f1)
	with open("labels.pkl", "rb") as f2:
		labels = pickle.load(f2)
	Test_data = np.transpose(np.loadtxt("TeData.txt",delimiter=','))
	Test_labels = np.loadtxt("TeLabel.txt",delimiter=',')
	return (w, labels, Test_data, Test_labels)

def identify(w, labels, img_vec):
	d_min = np.inf
	for i in range(w.shape[0]):
		for j in range(w.shape[1]):
			d = la.norm(w[i,j]-img_vec)
			if d <= d_min:
				d_min = d
				ii = i
				jj = j
	return int(labels[ii, jj])

def cal_accuracy(w, labels, Test_data, Test_labels):
	accu = 0
	for i in range(Test_data.shape[0]):
		img_vec = Test_data[i]
		ans = identify(w, labels, img_vec)
		if Test_labels[i] == ans:
			accu += 1
	return accu/len(Test_labels)


def main():
	w, labels, Test_data, Test_labels = load_data()
	accuracy = cal_accuracy(w, labels, Test_data, Test_labels)
	print(accuracy)

def visualize():
	w, labels, Test_data, Test_labels = load_data()
	fig, ax = plt.subplots()

	text = ax.text(0, -1, "")
	img = ax.imshow(np.transpose(Test_data[0].reshape([28,28])), cmap = "gray", interpolation = "nearest")

	def init():
		img.set_data([[]])
		text.set_text("ans = ")
	def data_gen():
		for i in range(Test_data.shape[0]):
			img_vec = Test_data[i]
			yield (img_vec, identify(w, labels, img_vec))
	def update(data):
		img_vec, n = data
		img.set_data(np.transpose(img_vec.reshape(28,28)))
		text.set_text("ans = {}".format(n))
		return (img,text)


	anim = animation.FuncAnimation(fig, update, data_gen, init_func=init, interval= 1000)
	plt.show()

if __name__ == "__main__":
	main()
	visualize()
