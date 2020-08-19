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


Activation Functions: 
 * Used to set boundaries to output values of a neuron
 * We know that inputs **x** have a weight **w** and a bias term attached to them **b**
    * Which means we have: x*w + b 
 * b is essentially an offset value which is a threshold that x*w has to overcome
 * So here lets say z = x*w + b 
 * Activation Functions are f(z)
 * For Binary Classification, you can use a simple step function where: 
    * if the output is greater than zero, we output one 
    * it its less than zero we output zero 
 * But really small changes aren't reflected 
 * Another activation functions:  
    * Hyperbolic Tangent : tanh(z)
        * outputs between -1 and 1 instead of 0 to 1 
    * Rectified Linear Unit : max(0,z)
        * very good performace, especially with vanishing gradient
        * we usually default to ReLU for activation functions

Multi Class Activation Functions: 
 * In a multi - class situtaion, the output will have multiple neurons
 * 2 main types of multi class situations: 
    * Non-Exclusive Class: 
        * A data point can have mulitple categories assigned to it
        * eg:  Photo which can have mulitple tag like beach, family, holiday, etc
    * Mutually Exclusive Class: 
        * Only one class per data point
        * eg:  Photos can be categorized as grayscale or color not both
 * To organize data in multiple classes, you need to have one output node per class 
 * We need to organize categories for each output layer : One-Hot Encoding (Dummy Variables)
 * We can use binary classification for each class, thus building out a matrix, thus creating dummy variables 
 * For Non Exclusive Classes: 
    * We perform the same idea, but the data points can have a value of one for multiple categories
 * Now we choose the best classification function: 
    * Non Exclusive Classes: Use Sigmoid/Logistic Function
    * Mutually Exclusive Classes: Use Softmax Function
        * Calculates the probibilities distribution of the event over _k_ different events
        * Probabilities of each target class over all target classes
        * Sum of all probabilites = 1 
        * The target class is chosen by the neuron with the highest probability

Cost/Error Functions: (y = true value, a = neuron prediction)
 * Used to compare our neural networks output to the true value
 * It has to be an AVERAGE 
    * so it can output of a single value 
    * then you can keep track of the loss/cost during training.
 * 'a' takes into account the activation function, weights and biases
 * Quadratic Cost Function: 
    * Looks kinda similar to root means square error formula 
    * punsishes really large errors because of the square in the formula 
    * also the square makes everything positive making it easier to compare 
 * Cost Fuction: C(W, B, S^r, E^r)
    * W - weight of neural networks 
    * B - neural network biases 
    * S^r - input of single training sample 
    * E^r - desired output of the training sample 
        * a(x) - hold info about weights and biases 
 * Larger the network, more complex the cost function
    * Since we have a lot of weights, we need to figure out which weights lead to the lowest cost 
    * what value of w minimizes C(w)
 *  The C(w) is going to be n dimensional, so we can't really plot the function.
    * We use **Gradient Descent** to solve this problem
        * You start off at one point in the cost function and find the slope at that point 
        * Move at the downward direction of the slope until you converge to zero which is the minimum
        * We also could change the step size, 
            * Smaller step sizes take longer to find the min 
            * Larger step sizes may overshoot 
        * These Step Sizes are also called the **Learning Rate**
* Learning Rates can be adjusted from larger to smaller for efficiency 
    * called **adaptive gradient descent**
* When we deal with n-dimensional vectors, the notation changs from derivative to **gradient** 
* For classification (especially multi-class) problems, instead of using quadratic functions, we use **cross entropy loss function**
    * This assumes that the model predicts a probability distribution for each class

Backpropagation: 
 * Now we try to figure out the relationship between the final cost function and the weights at the last layer 
   * we take the partial derivative of the cost function with respect to the weights and layer L 
 * However, the costs function is also affected by the bias along the network and that needs to be included here as well
   * we take the partial derivative of the cost function with respect to the bias terms and layer L 
 * We can now use the gradient to go back and adjust the weights and biases to minimize the error at the last layer L
   * Use the Hadamard Product (element by element multiplication)

* STEP 1: Use input _x_ to set activation function _a_ for the input layer 
   * z = x*w + b
   * a = sigmoid(z)
   * the resulting a feeds into the next layer 
* Step 2: For each layer, compute the _z_ and _a_ values 
* Step 3: Computer the error vector
   * Now we write a generalized error vector formula in terms of the error in the next layer 
* Step 4 : Backpropagate the error(check images for calculus)
   * When you apply the transpose weight matrix, think of it like you're oving the error backward through the network
   * The Hadamard Product helps us do that

Classification with Tensorflow:
 * Early Stopping: Used to automatically stop training based on a loss condition on the validation data in the model.fit() call
 * Dropout Layers: Layers that can be added to turn off neurons during training to prevent overfitting 
   * Each dropout layer will "drop" a user-defined percentage of neuron units in the previous layer every batch 
   * SO certain neurons don't have their weights and biases affeected 
