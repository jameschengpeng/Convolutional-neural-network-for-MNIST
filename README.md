# Convolutional-neural-network-for-MNIST
 I implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch). I also write my own code for convolutions (e.g., do not use SciPy's convolution function). The convolution network should have a single hidden layer with multiple channels. It should achieve at least 96% accuracy on the Test Set. 
This CNN has a single hidden layer with multiple channels. Since the depth
of pictures in MNIST are all 1, I used multiple filters of the same size to get the feature
maps. In general, the CNN was implemented in OOP style in which user can define the
picture size (height, width and the default depth is 1), the filter size (we only consider
square filters) as well as the number of filters. Then we compute the hidden layer, fully
connected layer and output layer and used backpropagation to minimize the objective
function. Here is the result.
