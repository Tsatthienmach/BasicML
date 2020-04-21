import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
	"""
	fit_intercept: boolean parameter , defaut: True
		True: add 1s of array input a zero column, which is relevant to adding a bias.
	"""
	def __init__(self, fit_intercept = True):
		self.fit_intercept = fit_intercept

	def _transform_to_bar(self, X):
		return np.column_stack((np.ones((X.shape[0])), X))

	def _calcu_A(self, X):
		return np.dot(X.T , X)

	def _calcu_b(self, X, y):
		return np.dot(X.T, y)

	def _calcu_w(self,A,b):
		w = np.dot(np.linalg.pinv(A) , b)
		return w.reshape(w.shape[0] , 1)

	def fit(self,X,y):
		X = np.asarray(X)
		y = np.asarray(y)
		if X.ndim == 1:
			X = X.reshape(X.shape[0] , 1)
		if y.ndim == 1:
			y = y.reshape(y.shape[0],1)
		elif y.ndim == 2:
			if y.shape[1] != 1:
				raise ValueError('The label must be one column array')
		else:
			raise ValueError('The label must be one column array')

		if self.fit_intercept == True:
			X = self._transform_to_bar(X)
		A = self._calcu_A(X)
		b = self._calcu_b(X,y)
		self.w = self._calcu_w(A,b)

	def predict(self, X):
		X = np.asarray(X)
		if X.ndim == 0:
			X = np.asarray([X])
			X.reshape(1,1)
		elif X.ndim == 1:
			X.reshape(X.shape , 1)
		else:
			raise ValueError('Too many dimensions are got.')

		if self.fit_intercept == True:
			X = self._transform_to_bar(X)
		return np.dot(X,self.w)

	def _coef(self):
		try:
			return self.w
		except AttributeError as error:
			print('Should fit model first')
			raise AttributeError(error)

	def plot_(self,X,y):
		if np.asarray(X).ndim == 2:
			if np.asarray(X).shape[1] > 1:
				print('Cant plot multiLinearRegression')
				return None
		if self.fit_intercept == True:
			X_draw = np.linspace(np.min(X) , np.max(X) , (np.max(X) - np.min(X))*2)
			X_draw = self._transform_to_bar(X_draw)
			plt.scatter(X,y,c='blue',label='Training data')
		else:
			X_draw = np.linspace(np.min(X[:,1]) , np.max(X[:,1]) , (np.max(X[:,1]) - np.min(X[:,1]))*2)
			plt.scatter(X[:,1], y, c='blue', label='Training data')
		try:
			y_draw = np.dot(X_draw , self.w)
		except AttributeError as error:
			print('Should fit model first')
			raise AttributeError(error)

		plt.plot(X_draw[:,1],y_draw,c='red', label='Linear')
		plt.legend()
		plt.show()


if __name__=='__main__':
	X= np.array([147,150,153,158,163,165,168,170,173,175,178,180,183])
	y= np.array([49,50,51,54,58,59,60,62,63,64,66,67,68])

	regr = LinearRegression(fit_intercept=True)
	regr.fit(X,y)
	# print(regr._coef())
	regr.plot_(X,y)