import numpy as np
from preprocessing import check_non_negative,check_X_y,check_X

class BornouliNB:
	def __init__(self, alpha=1.0):
		self.alpha = alpha

	def fit(self, X, y):
		X = check_non_negative(X)
		X,y = check_X_y(X,y)
		self.data_class = np.unique(y)
		class_probability = []
		self.class_data_probability = []
		for class_ in self.data_class:
			class_data = X[np.where(y == class_)]
			self.class_data_probability.append(class_data.shape[0] / X.shape[0])
			Nci = np.sum(class_data, axis=0)
			self.Dimensions = X.shape[1]
			Nc = self.Dimensions * self.alpha + np.sum(Nci)
			class_probability.append((Nci + self.alpha) / Nc)
		self.class_pro = np.asarray(class_probability)

	def predict(self, X):
		X = np.asarray(X)
		if X.ndim == 1:
			X = np.expand_dims(X, axis=0)
		class_probability = []
		# print(X)
		for point in range(X.shape[0]):
			point_probability = []
			for class_ in range(self.data_class.shape[0]):
				# init p = p_class * ...
				p_class_over_X = self.class_data_probability[class_]
				for d in range(self.Dimensions):
					if X[point,d] != 0:
						p_class_over_X *= (self.class_pro[class_][d])
				point_probability.append(p_class_over_X)
			point_probability = point_probability / (np.sum(point_probability))
			class_probability.append(point_probability)
		print(class_probability)
		postions = np.argmax(class_probability, axis=1)
		y = []
		for n in range(X.shape[0]):
			y.append(self.data_class[postions[n]])
		return y


class MultinomialNB:
	def __init__(self,alpha=1.0):
		self.alpha = alpha
	def fit(self,X,y):
		X = np.asarray(X)
		y = np.asarray(y)
		self.data_class = np.unique(y)
		class_probability = []
		self.class_data_probability = []
		for class_ in self.data_class:
			class_data = X[np.where(y==class_)]
			self.class_data_probability.append(class_data.shape[0]/X.shape[0])
			Nci = np.sum(class_data,axis=0)
			self.Dimensions = X.shape[1]
			Nc = self.Dimensions*self.alpha + np.sum(Nci)
			class_probability.append((Nci + self.alpha)/Nc)
		self.class_pro =  np.asarray(class_probability)

	def predict(self,X):
		X = np.asarray(X)
		if X.ndim  == 1:
			X = np.expand_dims(X, axis=0)
		class_probability = []
		# print(X)
		for point in range(X.shape[0]):
			point_probability = []
			for class_ in range(self.data_class.shape[0]):
				#init p = p_class * ...
				p_class_over_X = self.class_data_probability[class_]

				for d in range(self.Dimensions):
					p_class_over_X *= (self.class_pro[class_][d])**(X[point,d])
				point_probability.append(p_class_over_X)
			point_probability = point_probability/(np.sum(point_probability))
			class_probability.append(point_probability)
		print(class_probability)
		postions = np.argmax(class_probability, axis=1)
		y = []
		for n in range (X.shape[0]):
			y.append(self.data_class[postions[n]])
		return y

class GaussianNB:
	def __init__(self):
		pass

	def _data_per_class(self,X,y):
		self.class_count = np.unique(y , axis= 0)
		data = []
		self.prob_per_class =[]
		for class_ in self.class_count:
			data.append(X[np.where(class_==y)])
			self.prob_per_class.append(X[np.where(class_==y)].shape[0] / X.shape[0] )
		return data

	def _cal_mean_variance(self, data_per_class):
		mean_variance = []
		for class_ in range(self.class_count.shape[0]):
			data_per_class_ = data_per_class[class_]
			mean_ = np.mean(data_per_class_, axis=0)
			variance_ = np.var(data_per_class_, axis=0)
			mean_variance.append([mean_,variance_])
		return np.asarray(mean_variance)

	def fit(self,X,y):
		X,y = check_X_y(X,y)
		data_per_class = self._data_per_class(X,y)
		self.cal_mean_variance = self._cal_mean_variance(data_per_class)

	def _p_xi_c(self,x,mean_,var_):
		p = 1/np.sqrt(2*np.pi*var_) * np.exp(-1*(x-mean_)**2/(2*var_))
		return p

	def predict(self,X):
		X = check_X(X)
		n_class = []
		for n in range(X.shape[0]):
			p_class_ = []
			for class_ in range(self.class_count.shape[0]):
				p_ = self.prob_per_class[class_]
				for d in range(X.shape[1]):
					p_ *= self._p_xi_c(X[n,d],self.cal_mean_variance[class_,0,d],self.cal_mean_variance[class_,1,d])
				p_class_.append(p_)
			evidence = np.sum(p_class_)
			p_class_ = np.asarray(p_class_)/evidence
			n_class.append(self.class_count[np.argmax(p_class_)])
		return n_class

class NernouliNB:
	pass


if __name__ == '__main__':
	## Multinouli NB
	d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
	d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
	d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
	d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]
	# train data
	train_data = np.array([d1, d2, d3, d4])
	label = np.array(['B', 'B', 'B','N'])
	# test data
	d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
	d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
	d7 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
	d_test = np.row_stack((d5,d6,d7))

	naive = MultinomialNB()
	naive.fit(train_data, label)
	print(naive.predict(d_test))

	## Gaussian NB
	X = np.array([[6,180,12] ,[5.92,190,11] , [5.58 , 170,12],[5.92,165,10],
	              [5,100,6],[5.5,150,8] , [5.42,130,7],[5.75,150,9]])
	y = np.array(['male','male','male','male','female','female','female','female'])
	X_test = [[6,130,8],[6,180,12]]
	g = GaussianNB()
	g.fit(X,y)
	# print(X[np.where(y=='male')])
	# print('mmm' , np.var(X[np.where(y=='male')], axis=0))
	# dd = g._data_per_class(X,y)
	# print(g._cal_mean_variance(dd))
	# print(g.predict(X_test))

	## Bernouli NB
	d1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
	d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
	d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
	d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]
	# train data
	train_data = np.array([d1, d2, d3, d4])
	label = np.array(['B', 'B', 'B', 'N'])
	# test data
	d5 = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0]])
	d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
	d7 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
	d_test = np.row_stack((d5, d6, d7))

	bernb = BornouliNB()
	bernb.fit(train_data, label)
	print(bernb.predict(d_test))