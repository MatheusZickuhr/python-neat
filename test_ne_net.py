from python_neat.core.ga_neural_network.ne_neural_network import NeNeuralNetwork

net1 = NeNeuralNetwork(input_shape=(2,), output_size=2)
net2 = NeNeuralNetwork(input_shape=(2,), output_size=2)

c1, c2, = net1.crossover(net2)

print(c1.model.layers[0].input_shape)
