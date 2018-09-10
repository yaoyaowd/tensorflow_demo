import random
import math


class Neuron:
    def __init__(self, bias=0):
        self.bias = bias
        self.weights = []

    def get_output(self, inputs):
        self.inputs = inputs
        self.output = self.active(self.multiple())
        return self.output

    def multiple(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def active(self, value):
        return 1 / (1 + math.exp(-value))

    # Mean Square Error
    def error(self, target):
        return 0.5 * (target - self.output) ** 2

    def pd_error_wrt_output(self, target):
        return -(target - self.output)

    def pd_error_wrt_input(self, target):
        return self.pd_error_wrt_output(target) * self.output * (1 - self.output)

    def pd_input_wrt_weight(self, index):
        return self.inputs[index]


class Layer:
    def __init__(self, num_neurons, bias):
        self.bias = bias if bias else random.random()
        self.neurons = [Neuron(self.bias) for i in range(num_neurons)]

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for i in range(len(self.neurons))
            print(' Neuron:', i)
            for w in self.neurons[i].weights:
                print('  Weights:', w)
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        return [n.get_output(inputs) for n in self.neurons]

    def get_outputs(self):
        return [neuron.output for neuron in self.neurons]


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs,
                 hidden_layer_weights=None,
                 hidden_layer_bias=None,
                 output_layer_weights=None,
                 output_layer_bias=None):
        self.num_inputs = num_inputs
        self.hidden_layer = Layer(num_hidden, hidden_layer_bias)
        self.output_layer = Layer(num_outputs, output_layer_bias)
        self.init_weights(self.hidden_layer, hidden_layer_weights)
        self.init_weights(self.output_layer, output_layer_weights)

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def init_weights(self, layer, weights):
        weight_num = 0
        for h in range(len(layer.neurons)):
            for i in range(self.num_inputs):
                if not weights:
                    layer.neurons[h].weights.append(random.random())
                else:
                    layer.neurons[h].weights.append(weights[weight_num])
                weight_num += 1

    def feed_forward(self, inputs):
        hidden_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_outputs)

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].error(training_outputs[o])
        return total_error

    # Use online learning, updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        pd_errors_wrt_output = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output[o] = self.output_layer.neurons[o].pd_error_wrt_input(training_outputs[o])

        pd_error_wrt_hidden = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            d = 0
            for o in range(len(self.output_layer.neurons)):
                d += pd_errors_wrt_output[o] * self.output_layer.neurons[o].weights[h]
            pd_error_wrt_hidden[h] = d * self.hidden_layer.neurons[h].pd_error_wrt_input()

        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_output[o] * self.output_layer.neurons[o].pd_input_wrt_weight(w_ho)
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                pd_error_wrt_weight = pd_error_wrt_hidden[h] * self.hidden_layer.neurons[h].pd_input_wrt_weight(w_ih)
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight


nn = NeuralNetwork(2, 2, 2,
                   hidden_layer_weights=[0.15, 0.2, 0.25, 0.3],
                   hidden_layer_bias=0.35,
                   output_layer_weights=[0.4, 0.45, 0.5, 0.55],
                   output_layer_bias=0.6)
for i in range(10000):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))
