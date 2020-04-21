import numpy as np
from preprocessing import check_X_y,check_non_negative,check_X, check_binary

class  informationGain:
	def __init__(self):
		pass

	def Entropy(self,X,labels):
		E = 0
		N = X.shape[0]
		for label in self.label_unique:
			X_label = X[labels == label]
			n = X_label.shape[0]
			if n == 0:
				E+=0
			else:
				E += -1*(n/N)*np.log2(n/N)
		return E

	def fit(self,X,y):
		check_binary(X)
		check_binary(y)
		X,y = check_X_y(X,y)
		self.label_unique = np.unique(y)
		self.N = X.shape[0]
		self.feature_unique = np.unique(X)
		dimensions = [x for x in range(X.shape[1])]
		label_Entropy = self.Entropy(y,y)
		InforGain = []
		InforEntropy = []
		Leaf = []
		for d_ in dimensions:
			Entropy_dim = []
			Gain_dim = label_Entropy
			for feature in self.feature_unique:
				X_dim = X[:,d_]
				address_ = np.where(X_dim == feature)
				X_feature = X_dim[address_]
				y_feature = y[address_]
				n_feature = X_feature.shape[0]
				Entropy_ = self.Entropy(X_feature,y_feature)
				Entropy_dim.append(Entropy_)
				Gain_dim -= n_feature/self.N * Entropy_
				print(X_feature , '\t', y_feature)
				Leaf_feature = []
				for label in self.label_unique:
					print(label, '\t',np.where(y_feature == label))
					sum_label = X_feature[y_feature == label].shape[0]
					print(sum_label)
					Leaf_feature.append(sum_label)

			InforGain.append(Gain_dim)
			InforEntropy.append(Entropy_dim)
		self.DecisionTree = [x for _,x in sorted(zip(InforGain, InforEntropy), reverse=True)]
		self.root_sorted = [x for _,x in sorted(zip(InforGain, dimensions),reverse=True)]
		self.InforGain = sorted(InforGain, reverse=True)

	def predict(self,X):
		check_binary(X)
		X = check_X(X)
		if self.InforGain[0] == 1:
			for f_ in self.feature_unique:
				pass
		# return y

class GiniIndex:
	pass

if __name__=='__main__':

	X1 = [[1,1,1],[1,1,0],[0,0,1],[1,0,0]]
	X2 = [1,1,1]
	y1 = ['I','I','II',"II"]
	ig = informationGain()
	ig.fit(X1,y1)
	# print(ig.predict(X2))

	X2 = np.array([[4.8,3.4,1.9,0.2] ,
                  [5,3,1.6,0.2],
                  [5,3.4,1.6,0.4],
                  [5.2,3.5,1.5,0.2],
                  [5.2,3.4,1.4,0.2],
                  [4.7,3.2,1.6,0.2],
                  [4.8,3.1,1.6,0.2],
                  [5.4,3.4,1.5,0.4],
                  [7,3.2,4.7,1.4],
                  [6.4,3.2,4.5,1.5],
                  [6.9,3.1,4.9,1.5],
                  [5.5,2.3,4,1.3],
                  [6.5,2.8,4.6,1.5],
                  [5.7,2.8,4.5,1.3],
                  [6.3,3.3,4.7,1.6],
                  [4.9,2.4,3.3,1]])
	y2 = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])

