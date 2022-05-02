# Hazelnut

A simple module based machine learning library written with only NumPy and optionally CuPy for GPU acceleration

## Usage
### Start defining a Net

To create a neural net create a class off of Hazelnut.NN
```python
import Hazelnut

class Net(Hazelnut.NN):
    def __init__(self):
        super().__init__() #Make sure to include this
```

### Start adding layers to your net

To add layers, use `self.add()` and include a object from the modules folder.

Here's an example of a net with a convolution layer and a linear (fully connected) layer:

```python
import Hazelnut

class Net(Hazelnut.NN):
    def __init__(self):
        super().__init__()
        
        self.add(Hazelnut.modules.Conv(8, 3, mode='Same')) #Adding a 3x3 convolution layer with 8 filters
        self.add(Hazelnut.modules.ActivationFunction.ReLU()) #ReLU activation function
        self.add(Hazelnut.modules.Flatten()) #Flatten before going to fully connected layers
        self.add(Hazelnut.modules.Linear(10)) #Linear layer with 10 neurons
        self.add(Hazelnut.moudles.ActivationFunction.tanh()) #Hyperbolic tangent activation function
```

Alternatively, put all the layers as a list in `self.layout`

```python
import Hazelnut

class Net(Hazelnut.NN):
    def __init__(self):
        super().__init__()
        
        #This gives the same result as the one above
        self.layout = [
            Hazelnut.modules.Conv(8, 3, mode='Same')
            Hazelnut.modules.ActivationFunction.ReLU()
            Hazelnut.modules.Flatten()
            Hazelnut.modules.Linear(10)
            Hazelnut.moudles.ActivationFunction.tanh()
        ]
```

### Put in a loss function

Add a loss function by setting `self.loss`

For example:
```python
import Hazelnut

class Net(Hazelnut.NN):
    def __init__(self):
        super().__init__()
        ...

        self.loss = Hazelnut.modules.LossFunction.MSE() #Using mean squared error as the loss function
```

Implemented loss functions are in `Hazelnut.modules.LossFunction`

### Add in an optimizer

Use an optimizer by setting `self.optimizer`

```python
import Hazelnut

class Net(Hazelnut.NN):
    def __init__(self):
        super().__init__()
        ...

        self.optimizer = Hazelnut.modules.Optimizers.Adam(earning_rate=1e-6, beta1=0.9, beta2=0.99) #Using Adam as the optimizer
```

Implemented optimizers are in `Hazelnut.modules.Optimizers`

### CPU/GPU

Hazelnut runs on the CPU using NumPy by default but has the ability to run on GPU using CuPy.

To enable GPU, set `self.mode = 'gpu'`. Otherwise, set `set.mode = 'cpu'` or just ignore it!

```python
import Hazelnut

class Net(Hazelnut.NN):
    def __init__(self):
        super().__init__()
        ...

        self.mode = 'gpu' #Enabling GPU acceleration
```

### Build your net to finish!

Put `self.build(input_shape)` at the very end to initalize your net and it's ready!


```python
import Hazelnut

class Net(Hazelnut.NN):
    def __init__(self):
        super().__init__()
        ...

        self.build((3, 20, 30)) #Build the neural net for inputs with 3 channels, 20 height, and 30 width
```

### Run the net

Call the `forward` method to use the net 

```python
net = Net()
out = net.forward(inp)
```

Call the `train` method to train the net
loss, out = net.train(inp_batch, labels_batch)

## Example
Check out 
https://github.com/ecjiang016/MNIST-Hazelnut/settings
for an example on using Hazelnut
