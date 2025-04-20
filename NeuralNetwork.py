# For generating normal distributions
from numpy import random
# For Activation Functions
from enum import Enum
# Advanced math
import math

# This extended enumerated class provides better readability for activation funcitions
# It also holds operations which depeneds on the activation functions
class ActivationFunction(Enum):
    NONE = 0
    RELU = 2
    LEAKY_RELU = 4
    SIGMOID = 1
    TANH = 3

    # Applies the activation function to a given input, used to simplify other classes' calculation
    def calcActivationFunction(self, x: float) -> float:
        if self == ActivationFunction.RELU:
            return max(0, x)
        elif self == ActivationFunction.LEAKY_RELU:
            return max(0.01 * x, x)
        elif self == ActivationFunction.SIGMOID:
            return 1 / (1 + pow(math.e, -x))
        elif self == ActivationFunction.TANH:
            return math.tanh(x)
        return x

    # Applies the derivative of the function to a given input
    def calcDerivative(self, x: float) -> float:
        if self == ActivationFunction.RELU:
            return 0 if x <= 0 else 1
        elif self == ActivationFunction.LEAKY_RELU:
            return 0.01 if x <= 0 else 1
        elif self == ActivationFunction.SIGMOID:
            return x * (1 - x)
        elif self == ActivationFunction.TANH:
            return 1 - x * x
        return x


    # Returns the list-representation for the initial weights of a layer based on the activation function
    # This gives a better distribution for the initial weights
    def initWeights(self, numNeurons: int, numInputs: int) -> list[list[float]]:
        """Initialize weights based on activation function"""
        scale = 0.01
        if self in {ActivationFunction.SIGMOID, ActivationFunction.TANH}:
            scale = (2 / (numInputs + numNeurons)) ** 0.5
        if self in {ActivationFunction.RELU, ActivationFunction.LEAKY_RELU}:
            scale = (2 / numInputs) ** 0.5

        return [[float(random.normal(0, scale)) for i in range(numInputs)]
                for j in range(numNeurons)]


# This represents a list on n neurons and necessary operations, a layer is connected to k neurons in the previous layer
# Each neuron has a list of k weights corresponding to each neuron in the previous layer as well as a weight
# Contents: Weights[[,], [,], [,]...] Biases[ , , , ...]
class Layer:
    # Raises an exception if there is some error in the structure of a layer
    def checkValid(self):
        for weight in self.__weights:
            for w in weight:
                if type(w) != float and type(w) != int:
                    raise TypeError(f"Weight {w} is of type {type(w)} but should be not of type float")
        for bias in self.__biases:
            if type(bias) != float and type(bias) != int:
                raise TypeError(f"Bias {bias} is not of type float")
        if len(self.__weights) != len(self.__biases):
            raise (ValueError
                   (f"Weights and biases must have the same length: {len(self.__weights)} != {len(self.__biases)}"))

        for weight in self.__weights:
            if len(weight) != len(self.__weights[0]):
                raise (ValueError
                       (f"Weights must have the same length: {weight} does not match {len(self.__weights[0])}"))

    # Two ways to initialize a layer:
    # 1. The direct list of weights and biases
    # 2. A list of two elements for weights: number of neurons and the number of connections for each neuron
    # as well as a bias initializer
    def __init__(self, weights: list[list[float]] or list[int], biases: list[float] or int,
                 activationFunction: ActivationFunction = ActivationFunction.NONE):

        if type(activationFunction) != ActivationFunction:
            raise (TypeError
                   (f"activationFunction must be of type ActivationFunction but is {type(activationFunction)}"))

        self.activationFunction = activationFunction

        # [Num inputs, Num neurons]
        if len(weights) == 2 and type(weights[0]) == int and type(weights[1]) == int:
            weights = self.activationFunction.initWeights(weights[0], weights[1])

        # Init number
        if type(biases) == float or type(biases) == int:
            biases = [biases for _ in range(len(weights))]

        self.__weights = weights
        self.__biases = biases

        self.checkValid()

    # Readable (not really) representation of a layer for debugging
    def __str__(self):
        return f"Weights: {self.__weights}, Biases: {self.__biases}, Activation Function: {self.activationFunction}"

    # Portable representation of layer
    def __repr__(self):
        return f"{self.__weights}, {self.__biases}, {self.activationFunction}"

    def weights(self) -> list[[float]]:
        return self.__weights

    def biases(self) -> list[float]:
        return self.__biases

    def setAllWeights(self, neuron: int, weights: list[float]) -> None:
        self.__weights[neuron] = weights.copy()
        self.checkValid()

    def setBiases(self, biases: list[float]) -> None:
        self.__biases = biases.copy()
        self.checkValid()

    def setWeights(self, neuron: int, weight: list[float]) -> None:
        self.__weights[neuron] = weight.copy()
        self.checkValid()

    def setBias(self, neuron: int, bias: float) -> None:
        self.__biases[neuron] = bias
        self.checkValid()

    def setWeight(self, neuron: int, input: int, weight: float) -> None:
        self.__weights[neuron][input] = weight
        self.checkValid()

    def changeBias(self, neuron: int, bias: float) -> None:
        self.__biases[neuron] += bias
        self.checkValid()

    def changeWeight(self, neuron: int, input: int, weight: float) -> None:
        self.__weights[neuron][input] += weight
        self.checkValid()

    def numNeurons(self) -> int:
        return len(self.__weights)

    def numInputs(self) -> int:
        return len(self.__weights[0])

    def getOutputs(self, inputs: list[float]) -> list[float]:
        res = []

        for i in range(self.numNeurons()):
            res.append(0)
            for j in range(self.numInputs()):
                res[i] += self.__weights[i][j] * inputs[j]

            res[i] += self.__biases[i]

            if self.activationFunction == ActivationFunction.RELU:
                res[i] = max(0, res[i])
            elif self.activationFunction == ActivationFunction.LEAKY_RELU:
                res[i] = max(0.01 * res[i], res[i])
            elif self.activationFunction == ActivationFunction.SIGMOID:
                res[i] = 1 / (1 + pow(math.e, -res[i]))
            elif self.activationFunction == ActivationFunction.TANH:

                res[i] = math.tanh(res[i])
        return res

class NeuralNetwork:
    def numLayers(self) -> int:
        return len(self.__layers)

    def layers(self) -> list[Layer]:
        return self.__layers

    def layer(self, index: int) -> Layer:
        return self.__layers[index]

    # Can the neural network be used
    # Must have
    def checkValid(self):
        # At least one layer
        if self.numLayers() == 0:
            raise Exception("NeuralNetwork must have at least one layer")

        # Each layer must have the same number of inputs as the number of neurons in the previous layer
        for i in range(self.numLayers() - 1):
            if self.layers()[i].numNeurons() != self.layers()[i+1].numInputs():
                raise ValueError(f"Layer {i} has {self.layers()[i].numNeurons()} neurons(s) but "
                                 f"layer {i+1} has {self.layers()[i+1].numInputs()} neuron(s)")

        # All weights and biases must be of type double
        for layer in self.layers():
            for weight in layer.weights():
                for w in weight:
                    if type(w) != float and type(w) != int:
                        raise TypeError(f"Weight {w} is not of type float")
            for bias in layer.biases():
                if type(bias) != float and type(bias) != int:
                    raise TypeError(f"Bias {bias} is not of type float")

        return True

    def __init__(self, layers: list):
        self.__layers = []

        # Check that each layer is a Layer
        for l in layers:
            if type(l) == Layer:
                self.__layers.append(l)
            else:
                try:
                    self.__layers.append(Layer(l[0], l[1], l[2]))
                except:
                    raise TypeError(f"{l} is not a Layer or a list of weights and biases")

        try:
            self.checkValid()
        except Exception as e:
            raise e


    def __str__(self):
        return '\n'.join(f"{str(layer)}" for layer in self.__layers)


    #Exportable
    def __repr__(self):
        return  '[[' + ', ['.join(layer.__repr__() + ']' for layer in self.__layers) + ']'

    def numInputs(self) -> int:
        return self.layers()[0].numInputs()

    def numOutputs(self) -> int:
        return self.layers()[-1].numNeurons()

    def forwardFeed(self, input: list[float]) -> list[list[float]]:
        if len(input) != self.layers()[0].numInputs():
            raise ValueError(f"Input must have {self.layers()[0].numInputs()} inputs but has {len(input)}")

        res = [input.copy()]

        for layer in self.layers():
            res.append(layer.getOutputs(res[-1]))

        return res

    def backPropagate(self, input: list[float], target: list[float], learningRate: float):
        # Stops the gradient from increasing too quickly
        max_norm = 1

        res = self.forwardFeed(input)
        error = [[res[-1][i] - target[i]  for i in range(len(target))]]

        # Calculate errors for each layer
        for i in range(self.numLayers() - 2, -1, -1):
            error.insert(0, [])
            for j in range(self.layers()[i].numNeurons()):
                error[0].append(0)
                for k in range(self.layers()[i + 1].numNeurons()):
                    error[0][j] += self.layers()[i + 1].weights()[k][j] * error[1][k]
                error[0][j] *= self.layers()[i].activationFunction.calcDerivative(res[i + 1][j])

        # Update weights and biases
        for i in range(self.numLayers()):
            for j in range(self.layers()[i].numNeurons()):
                # Clip error to max_norm
                if abs(error[i][j]) > max_norm:
                    error[i][j] = max_norm if error[i][j] > 0 else -max_norm

                # Update weights
                for k in range(self.layers()[i].numInputs()):
                    delta = learningRate * error[i][j] * res[i][k]
                    self.layers()[i].changeWeight(j, k, -delta)

                # Update bias
                delta_bias = learningRate * error[i][j]
                self.layers()[i].changeBias(j, -delta_bias)

        return error

    @staticmethod
    def cost(output: list[float], target: list[float]) -> float:
        if len(output) != len(target):
            raise ValueError("Output and target must have the same length")

        cost = 0
        for i in range(len(output)):
            cost += pow(output[i] - target[i], 2)

        return cost

    def train(self, fileName: str, epochs: int, learningRate: float):
        with open(fileName, 'r') as f:
            for i in range(epochs):
                intput = [float(x) for x in f.readline().split(' ')]
                expected = [float(x) for x in f.readline().split(' ')]
                self.backPropagate(intput, expected, learningRate)


    def test(self, fileName: str, numTests: int) -> float:
        with open(fileName, 'r') as f:
            totalCost = 0
            for i in range(numTests):
                input = [float(x) for x in f.readline().split(' ')]
                expected = [float(x) for x in f.readline().split(' ')]
                output = self.forwardFeed(input)[-1]
                cost = NeuralNetwork.cost(output, expected)
                totalCost += cost
                #print(input, expected, output, cost)
                print(f'({input[0]}, {output[0]}) ')

        print(self.__repr__())
        print(f"Average cost: {totalCost / numTests}")

        return totalCost / numTests

    @staticmethod
    def generateData(fileName: str, numTests: int, min, max, func):
        with open(fileName, 'w') as f:
            for i in range(numTests):
                num = random.uniform(min, max)
                f.write(f"{num}\n{func(num)}\n")