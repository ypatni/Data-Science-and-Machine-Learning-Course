Perceptron Model: 
 * We're basically trying to figure out how biological neurons works to have the computer atrifically mimic natural intelligence
 * A perceptron was a form of neural network (even Google Translate uses )
 * Imagine two datapoints, x1 and x2, and inside the perceptron there is some function f(x) that gives the output (y)
 * We can also add adjustable weights (w1 and w2) to add to each input of x to get a correct value of y 
 * We can also add bias terms (b) to account for zero values so that the inputs are multiplied by the weigths and added to the bias 
 * We can then expand this to create a generalization from i = 1 to i = n

Neural Networks: 
 * In most cases a single perceptron model will never be enough to learn systems, 
 * so we can create a multi layered perceptron model in order to accomodate fro this - artificial neural network
 * To build a network of perceptrons we just connect the different layers of perceptrons using a multi-layer perceptron model 
 * So we have a vertical layer of neurons, and take their outputs and feed them to the next layers as their inputs 
 * This allows the network to learn about interactions and features 
 * The first layer is the input layer and usually directly receives the data 
 * The last layer is the output layer and can be more than one neuron 
 * Layers in between are the hidden layers and are difficult to interpret 
    * because of their interconnectivity and distance from input and output
 *  A DEEP neural network has 2 or more hidden layers 
 * This can be used to approximate any continuous function (Universal Approximation Theorem)

