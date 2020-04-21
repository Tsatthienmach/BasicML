import numpy as np

def cdist_(subtrac_array, metric , dimension = 2, p = 2):
	if dimension == 1:
		if metric == 'euclidean':
			return np.sqrt(np.sum(subtrac_array ** 2))
		if metric == 'manhattan':
			return np.sum(np.abs(subtrac_array))
		if metric == 'p-norm':
			return (np.sum( np.abs(subtrac_array)**p  )) ** (1 / p)
	if dimension == 2:
		if metric == 'euclidean':
			return np.sqrt(np.sum(subtrac_array ** 2 , axis = 1))
		if metric == 'manhattan':
			return np.sum(np.abs(subtrac_array) , axis = 1)
		if metric == 'p-norm':
			return (np.sum( np.abs(subtrac_array) ** p , axis = 1)) ** (1/p)

def cdist(XA, XB, metric='euclidean', p=2):
	if metric not in ['euclidean' , 'manhattan' , 'p-norm']:
		raise ValueError('metric expects one of [euclidean,manhattan,p-norm], but got {}'.format(metric))
	XA = np.asarray(XA)
	XB = np.asarray(XB)
	if XA.ndim == 1:
		if XB.ndim != 1:
			raise ValueError('The second array expect 1 dimension, but got {} dimensions'.format(XB.ndim))
		else:
			if XA.shape[0] != XB.shape[0]:
				raise ValueError('The length of second Vector expect {}, but got {}'.format(XA.shape[0], XB.shape[0]))
			subtrac_array = XA - XB
			return cdist_(subtrac_array, metric, XA.ndim, p)

	if XA.ndim == 2:
		if XB.ndim == 1:
			if XA.shape[1] != XB.shape[0]:
				raise ValueError('The column-dimensions of 2 arrays must be the same')
			else:
				subtrac_array = XA - XB
				return cdist_(subtrac_array, metric, XA.ndim,p)

		if XB.ndim == 2:
			if XA.shape[1] != XB.shape[1]:
				raise ValueError('The column-dimensions of 2 arrays must be the same')
			else:
				output = []
				for i in range(XB.shape[0]):
					subtrac_array = XA - XB[i,:]
					output.append(cdist_(subtrac_array, metric, XA.ndim , p))
				return np.asarray(output).transpose(1,0)
			
		if XB.ndim > 2:
			raise  ValueError('The dimensions of second array must be less than 3')

	if XA.ndim > 2:
		raise ValueError('The dimensions of first array must be less than 3')


if __name__=='__main__':
	a = np.array([[1,1] ])
	b = np.array([[1,1] , [0,0] , [7,6]])
	print(cdist(a,b))
