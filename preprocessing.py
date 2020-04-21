import numpy as np

def check_non_negative(X):
	X = np.asarray(X)
	if (X<0).any() :
		raise ValueError('There are negative values in array')
	else:
		return X

def check_binary(X):
	X = np.asarray(X)
	if len(np.unique(X)) not in [1,2]:
		raise ValueError('Array is not binary array')


def check_X_y( X, y):
	X = np.asarray(X)
	y = np.asarray(y)
	if X.ndim == 2:
		pass
	else:
		raise ValueError('Expect 2-d, but got {}-d'.format(X.ndim))

	if y.ndim > 1:
		raise ValueError('Expect 1 or 2-d, but got {}-d'.format(y.ndim))
	if X.shape[0] == y.shape[0]:
		return X, y
	else:
		raise ValueError('Expect same length of Input, label data')


def check_X( X):
	X = np.asarray(X)
	if X.ndim == 1:
		X = np.expand_dims(X, axis=0)
	elif X.ndim == 2:
		pass
	else:
		raise ValueError('Expect same length of Input, label data')
	return X

if __name__ == '__main__':
	a = np.array([[1,1],[1,1], [1,0]])
	check_binary(a)