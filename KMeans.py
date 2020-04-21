import numpy as np
from distance import cdist
import matplotlib.pyplot as plt
import sys
# %matplotlib notebook

class KMeans:
	def __init__(self, n_clusters = 3):
		self.n_clusters = n_clusters
	def _center_init(self,X,random_state):
		if random_state not in ['random','kmean++']:
			raise ValueError('Random state expected in {random,kmean++}, but got{}'.format(random_state))
		if random_state == 'random':
			centroids = X[np.random.choice(X.shape[0],self.n_clusters)]
		else:
			centroids = []
			centroids.append(X[np.random.randint(X.shape[0]),:])
			for next_centroid in range(self.n_clusters - 1):
				dist_ = cdist(X,centroids)
				dist_min = np.min(dist_,axis=1)
				centroids.append(X[np.argmax(dist_min), :])
			centroids = np.asarray(centroids)
		return centroids

	def _labeling(self,X,centroids):
		y = np.argmin(cdist(X,centroids) , axis=1)
		return y
	def _updateCanters(self,X,y, centroids):
		new_centroids = []
		for i in range(self.n_clusters):
			new_centroids.append(np.mean(X[np.where(y==i)], axis=0))
		return np.asarray(new_centroids)

	def fit(self,X,random_state='kmean++', plot_state =  False , max_iteration = 300):
		centroids = self._center_init(X,random_state)
		epoch = 0
		y = np.zeros(X.shape[0])
		fig = plt.figure(figsize =(8,8))
		axes = fig.add_subplot(111)
		while True:
			y = self._labeling(X,centroids)
			new_centroids = self._updateCanters(X,y,centroids)
			if set([tuple(a) for a in new_centroids]) == set([tuple(a) for a in centroids]) or epoch >= max_iteration:
				break
			else:
				centroids = new_centroids
				if plot_state == True:
					self._plot(fig,axes,X,y,centroids,epoch)
				epoch += 1
		self.centroids = centroids
		plt.close()
		return  self.centroids, y, epoch

	def predict(self,X,plot_state = True):
		X = np.asarray(X)
		if X.ndim == 1:
			X = np.expand_dims(X,axis=0)
		elif X.ndim == 2:
			pass
		else:
			raise ValueError('Array have so many dimensions')
		dist_ = cdist(X,self.centroids)
		y = np.argmin(dist_, axis=1)
		if plot_state == True:
			for i in range(self.n_clusters):
				plt.scatter(self.centroids[i, 0], self.centroids[i, 1], s=300 , label = str(i))
				X_draw = X[np.where(y==i)]
				if X_draw.shape[0] != 0:
					plt.scatter(X_draw[:,0] ,X_draw[:,1] , s = 50 , label = str(i))
			plt.legend()
			plt.show()
		return y

	def _plot(self,fig,axes,X,y,centroids,epoch):
		axes.cla()
		axes.set_title('Epoch: {}'.format(epoch))
		for i in range(self.n_clusters):
			X_draw = X[np.where(y==i)]
			axes.scatter(X_draw[:,0] , X_draw[:,1])
		axes.scatter(centroids[:, 0], centroids[:, 1], c='black', s=80)
		fig.canvas.draw()


class clusters_optimal:
	def __init__(self, X, k, random_state='kmean++', plot_state=False):
		self.X = np.asarray(X)
		self.random_state = random_state
		self.plot_state = plot_state
		if type(k) == int and k >= 3:
			self.k = [k for k in range(2, k + 1)]
		elif type(k) == list:
			self.k = k
		else:
			raise ValueError('Cluster k must be interger which is greater than 2 or list')

	def _WSS(self, k_index):
		model = KMeans(n_clusters=k_index)
		centroids, y, epoch = model.fit(self.X,random_state=self.random_state, plot_state=self.plot_state)
		wss_k = 0
		for centroid_index in range(centroids.shape[0]):
			cluster_data = self.X[np.where(y == centroid_index)]
			dis = cdist(cluster_data, [centroids[centroid_index]])
			sum_cluster = np.sum(dis)
			wss_k = wss_k + sum_cluster
		return wss_k, epoch

	def distortion(self):  # label: y , data: X , centroids
		distortion_out = []
		for k_index in self.k:
			wss_k, epoch = self._WSS(k_index)
			distortion_out.append([wss_k / self.X.shape[0], epoch])
		self.distortion_out = np.asarray(distortion_out)
		return self.distortion_out

	def inertia(self):
		inertia_out = []
		for k_index in self.k:
			wss_k, epoch = self._WSS(k_index)
			inertia_out.append([wss_k, epoch])
		self.inertia_out = np.asarray(inertia_out)
		return self.inertia_out

	def _a(self, dist_all_data_new, y, j):
		C_i = self.X[np.where(y == j)].shape[0]
		return 1 / (C_i - 1) * dist_all_data_new[j]

	def _b(self, dist_all_data_new, y, j):
		dist_all_data_new_temp = dist_all_data_new
		dist_all_data_new_temp[j] = sys.maxsize
		second_min = np.argmin(dist_all_data_new_temp)
		C_j = self.X[np.where(y == second_min)].shape[0]
		return 1 / C_j * dist_all_data_new[second_min]

	def _s(self, a_i, b_i):
		return (b_i - a_i) / (max(a_i, b_i))

	def silhouette(self):
		print(self.k)
		silhouette_out = []
		for k_index in self.k:
			model = KMeans(n_clusters=k_index)
			centroids, y, epoch = model.fit(self.X, random_state=self.random_state, plot_state=self.plot_state)
			dist_all_data = cdist(self.X, self.X)
			dist_all_data_new = np.zeros((self.X.shape[0], centroids.shape[0]))
			for i in range(self.X.shape[0]):
				for j in range(centroids.shape[0]):
					dist_all_data_new[i, j] = np.sum(dist_all_data[i, np.where(y == j)])
			s_k = []
			for i in range(self.X.shape[0]):
				j = int(y[i])
				a_i = self._a(dist_all_data_new[i], y, j)
				b_i = self._b(dist_all_data_new[i], y, j)
				s_i = self._s(a_i, b_i)
				s_k.append(s_i)
			silhouette_out.append([sum(s_k) / self.X.shape[0], epoch])
		self.silhouette_out = np.asarray(silhouette_out)
		return self.silhouette_out

	def plot_(self, k_clusters):
		f = plt.figure(figsize=(8, 8))
		ax1 = f.add_subplot(111)
		ax1.set_xlabel('k_cluster')
		ax1.set_ylabel('value')
		ax1.plot(self.k, k_clusters[:, 0], color='blue')
		ax2 = ax1.twinx()
		ax2.set_ylabel('epoch')
		ax2.plot(self.k, k_clusters[:, 1], color='green')
		plt.show()

def data_create():
	means = [[1, 5], [5, 2], [10, 9], [14, 15], [20, -4], [-5, -12]]
	cov = [[8, 0], [0, 8]]
	N = 300
	K = 5
	X = np.random.multivariate_normal(means[0], cov, N)
	for i in range(1, len(means)):
		X1 = np.random.multivariate_normal(means[i], cov, N)
		X = np.concatenate((X, X1), axis=0)
	return X

if __name__=='__main__':
	X = data_create()
	n_clusters = 5
# Test KMeans
	# kmeans = KMeans(n_clusters)
	# centroids , y, epoch = kmeans.fit(X)
	#
	# X_test = [[1,2], [6,8] , [4,5] , [0,3] , [-34,6]]
	# y_test = kmeans.predict(X_test)
	# print(y_test)
# Test Cluster_Optimal
# 	opti = clusters_optimal(X,10)
# 	distor = opti.inertia()
	# inertia = opti.inertia()
	# silhoue = opti.silhouette()
	# opti.plot_(distor)
