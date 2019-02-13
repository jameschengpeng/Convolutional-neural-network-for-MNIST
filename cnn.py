#Name: Peng Cheng
#UIN: 674792652
import numpy as np
import h5py
import time
import copy
from random import randint

def prod_sum(x,K,I,J,d,filter_height,filter_width): #means the dth kernel, top left is x[I][J]; K is 3 dimensional
	submatrix_x = x[I:I+filter_height, J:J+filter_width]
	summation =  np.sum(submatrix_x * K[:,:,d])
	return summation

def convolution(Filter, x, num_filter, height, width, filter_height, filter_width):
	Z = np.random.randn(height - filter_height + 1, width - filter_width + 1, num_filter)
	for d in range(num_filter):
		for I in range(height - filter_height + 1):
			for J in range(width - filter_width + 1):
				Z[I][J][d] = prod_sum(x, Filter, I, J, d, filter_height, filter_width)
	return Z

class single_hidden_multi_channel_cnn:
	def __init__(self, height, width, output_size, kernel_side, num_kernel, activation, iteration, learningRate):
		self.height = height
		self.width = width
		self.output_size = output_size
		self.kernel_side = kernel_side
		self.num_kernel = num_kernel
		self.activation = activation
		self.iteration = iteration
		self.learningRate = learningRate

	def get_W(self):
		self.W = np.random.randn(self.output_size, self.height - self.kernel_side + 1,
								self.width - self.kernel_side + 1, self.num_kernel) / np.sqrt((self.height - self.kernel_side + 1)*(self.width - self.kernel_side + 1)*(self.num_kernel))

	def get_K(self):
		self.K = np.random.randn(self.kernel_side, self.kernel_side, self.num_kernel)

	def get_b(self):
		self.b = np.random.randn(self.output_size, 1)

	def get_Z(self, x): # Z should be (height-kernel_side+1)*(width-kernel_side+1)*num_kernel
		self.Z = convolution(self.K, x, self.num_kernel, self.height, self.width, self.kernel_side, self.kernel_side)

	def get_H(self): # H should be (height-kernel_side+1)*(width-kernel_side+1)*num_kernel
		self.H = self.Z
		if self.activation == 'relu':
			for d in range(self.num_kernel):
				for i in range(self.height - self.kernel_side + 1):
					for j in range(self.width - self.kernel_side + 1):
						if self.Z[i][j][d] < 0:
							self.H[i][j][d] = 0
		elif self.activation == 'tanh':
			self.H = np.tanh(self.H)
		elif self.activation == 'sigmoid':
			for d in range(self.num_kernel):
				for i in range(self.height - self.kernel_side + 1):
					for j in range(self.width - self.kernel_side + 1):
						element = self.Z[i][j][d]
						self.H[i][j][d] = np.exp(element)/(1+np.exp(element))

	def get_U(self):
		self.U = [0 for k in range(self.output_size)]
		for k in range(self.output_size):
			dim = (self.height-self.kernel_side+1)*(self.width-self.kernel_side+1)*self.num_kernel
			Wk = self.W[k].reshape(1, dim)
			Uk = np.matmul(Wk, self.H.reshape(dim, 1))[0][0] + self.b[k] # indexing to get the number rather than a numpy array
			self.U[k] = Uk
		
		self.U = np.array(self.U).reshape(self.output_size, 1)

	def get_f(self):
		total = 0
		for entry in self.U:
			total += np.exp(entry)
		processed = []
		for entry in self.U:
			processed.append(np.exp(entry)/total)
		self.f = np.array(processed).reshape(self.output_size, 1)

	def partial_rho_partial_U(self, y):
		indicator = [0 for i in range(self.output_size)]
		indicator[y] = 1
		indicator = np.array(indicator).reshape(self.output_size, 1)
		self.rho_partial_U = -(indicator - self.f)

	def get_delta(self):
		self.delta = np.zeros((self.height - self.kernel_side + 1, self.width - self.kernel_side + 1, self.num_kernel))
		for k in range(self.output_size):
			self.delta += self.rho_partial_U[k][0] * self.W[k]

	def sigma_prime(self):
		self.sigma_prime_Z = self.Z
		if self.activation == 'tanh':
			self.sigma_prime_Z = 1 - np.tanh(self.Z) ** 2
		elif self.activation == 'relu':
			for d in range(self.num_kernel):
				for i in range(self.height - self.kernel_side + 1):
					for j in range(self.width - self.kernel_side + 1):
						if self.Z[i][j][d] > 0:
							self.sigma_prime_Z[i][j][d] = 1
						else:
							self.sigma_prime_Z[i][j][d] = 0
		elif self.activation == 'sigmoid':
			for d in range(self.num_kernel):
				for i in range(self.height - self.kernel_side + 1):
					for j in range(self.width - self.kernel_side + 1):
						element = self.Z[i][j][k]
						sigma = np.exp(element)/(1+np.exp(element))
						self.sigma_prime_Z[i][j][d] = sigma*(1 - sigma)

# training function
def train(train_x, train_y, theta): #model is a single_hidden_multi_channel_cnn object
	indexSet = np.random.choice(60000, theta.iteration, replace = True)
	for l in range(theta.iteration):
		index = indexSet[l]
		x = train_x[index]
		y = train_y[index]
		x = x.reshape(theta.height, theta.width)
		if l == 0:
			theta.get_W()
			theta.get_K()
			theta.get_b()
		theta.get_Z(x)
		theta.get_H()
		theta.get_U()
		theta.get_f()
		theta.partial_rho_partial_U(y)
		theta.get_delta()
		theta.sigma_prime()
		# update theta.b theta.W theta.K
		theta.b -= theta.learningRate * theta.rho_partial_U
		for k in range(theta.output_size):
			theta.W[k] -= theta.learningRate * theta.rho_partial_U[k] * theta.H
		theta.K -= theta.learningRate * convolution(theta.sigma_prime_Z * theta.delta, x, theta.num_kernel, 
													theta.height, theta.width, theta.height - theta.kernel_side + 1,
													theta.height - theta.kernel_side + 1)
		print("Iteration: " + str(l+1))

def softmax(x):
	total = 0
	for i in x:
		total += np.exp(i[0])
	processed_x = []
	for i in x:
		processed_x.append(np.exp(i[0])/total)
	return np.array(processed_x)

def forward(theta, x):
	Z = convolution(theta.K, x, theta.num_kernel, theta.height, theta.width, theta.kernel_side, theta.kernel_side)
	if theta.activation == 'tanh':
		H = np.tanh(Z)
	elif theta.activation == 'relu':
		H = Z
		for d in range(theta.num_kernel):
			for i in range(theta.height - theta.kernel_side + 1):
				for j in range(theta.width - theta.kernel_side + 1):
					if Z[i][j][d] < 0:
						H[i][j][d] = 0
	elif theta.activation == 'sigmoid':
		H = Z
		for d in range(theta.num_kernel):
			for i in range(theta.height - theta.kernel_side + 1):
				for j in range(theta.width - theta.kernel_side + 1):
					element = Z[i][j][d]
					H[i][j][d] = np.exp(element) / (1 + np.exp(element))
	U = np.zeros((theta.output_size, 1))
	for k in range(theta.output_size):
		dim = (theta.height-theta.kernel_side+1)*(theta.width-theta.kernel_side+1)*theta.num_kernel
		Wk = theta.W[k].reshape(1, dim)
		Uk = np.matmul(Wk, H.reshape(dim, 1))[0][0] + theta.b[k] # indexing to get the number rather than a numpy array
		U[k] = Uk
	f = softmax(U)
	return f


#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )

MNIST_data.close()
# some parameters
height = 28
width = 28
output_size = 10
kernel_side = 3
num_kernel = 5
activation = 'relu'
iteration = 100000
learningRate = 0.008


#train here
theta = single_hidden_multi_channel_cnn(height,width,output_size,kernel_side,num_kernel,activation,iteration,learningRate)
train(x_train, y_train, theta)
#test accuracy
total_correct = 0
for n in range(len(x_test)):
	y = y_test[n]
	x = x_test[n][:].reshape(28,28)
	p = forward(theta, x)
	prediction = np.argmax(p)
	if (prediction == y):
		total_correct += 1
	print("Test completed: " + str(n+1))
accuracy = total_correct/np.float(len(x_test) )
print(accuracy)