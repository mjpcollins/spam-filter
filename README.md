# spam-filter
Spam Filter for AI MSc with Bath University.

This spam filter is built with a fully connected Neural Network
with 3 layers.

The input layer has 54 nodes, the hidden layer has 3 nodes, 
and the output layer is a single node.

## Theory

Artificial Neural Networks (ANN) take Gradient Descent, introduce it
to Linear Algebra, and then let the magic happen.

Many people have attempted to explain how ANN produce 
exceptional classification models from algebra, but I believe 
these two are the best explanations out there.

- [3Blue1Brown's four YouTube videos on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Rafay Kahn's exceptionally detailed tutorial on the precise mathematics of ANNs](https://pub.towardsai.net/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0)

Rafay's tutorial also provided the numbers used in the unittests 
for the NN model I have created.

Generally speaking, a Neural Network's training operation has two phases:
1. Forward propagation
2. Backward propagation

### Forward Propagation

This section occurs by taking the inputs and multiplying them
by the weights. The result of this multiplication are added to 
a bias and then normalised with a Sigmoid function.

> Z = inputs &#8226; weights + bias
> 
> activation = sigmoid(Z)

As we have 54 inputs, then  3 hidden nodes, and finally 1 output node 
the activation is propagated across the layers and matrices are used.

> Z_hidden = inputs_matrix &#8226; hidden_weights_matrix + hidden_bias
> 
> activation_hidden = vectorised_sigmoid(Z_hidden)
> 
> Z_output = activation_hidden &#8226; output_weights_matrix + output_bias
> 
> activation_output = vectorised_sigmoid(Z_output)

### Backward Propagation

It is advised that you watch 3Blue1Brown's video for a clear explanation.

In essence, at this stage we take our ANN's prediction and compare it to the expected value 
then make a change to the weights and biases to bring the prediction closer to the expected value.

The direction of the change is calculated by working out the dCost/dWeight and dCost/dBias for
all weights and biases in the network.

To dive into the mathematics of this, I recommend reading through Rafay Kahn's tutorial 
as there is not much room here to explain them sufficiently.

## This Neural Network

The neural network I have created has been trained on some pre-prepared
"spam" data. The first column of the dataset (located in the data folder) are
the labels. The following columns are the features.

I found that the model trained to a reasonable standard (about 93% accurate) 
with the following settings:
- 5000 epochs
- Two batches of 500 training samples within each epoch
- One hidden layer with three nodes
- Learning rate of 1

These were chosen by trial and error. 
