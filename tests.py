import unittest

import numpy as np

from python_ne.core.ga import random_probability_selection
from python_ne.core.neural_network.dense_layer import DenseLayer
from python_ne.core.neural_network.neural_network import NeuralNetwork


class NeuralNetworkTest(unittest.TestCase):

    def test_initialization_on_initialize_call(self):
        nn = NeuralNetwork()
        nn.add(DenseLayer(units=256, activation='sigmoid', input_shape=(1,)))
        nn.add(DenseLayer(units=64, activation='sigmoid'))
        nn.add(DenseLayer(units=256, activation='sigmoid'))
        nn.add(DenseLayer(units=64, activation='sigmoid'))
        nn.initialize()

        nn.predict(np.array([1]))

        for layer in nn.layers:
            layer_weights = layer.get_weights()
            self.assertEqual(type(layer_weights[0]), np.ndarray)
            self.assertEqual(type(layer_weights[1]), np.ndarray)

    def test_initialization_on_predict_called(self):
        nn = NeuralNetwork()
        nn.add(DenseLayer(units=2, activation='sigmoid', input_shape=(1,)))
        nn.add(DenseLayer(units=2, activation='sigmoid'))
        nn.predict(np.array([1]))

        for layer in nn.layers:
            layer_weights = layer.get_weights()
            self.assertEqual(type(layer_weights[0]), np.ndarray)
            self.assertEqual(type(layer_weights[1]), np.ndarray)

    def test_set_weights_constructor(self):
        layer1_weights = np.array([
            [.5, .3]
        ])
        layer1_bias = np.array([-.1, -.6])

        layer2_weights = np.array([
            [.1, .4],
            [-.7, .2],
        ])
        layer2_bias = np.array([-.9, -.5])

        nn = NeuralNetwork()
        nn.add(DenseLayer(units=2, activation='sigmoid', input_shape=(1,),
                          weights=(layer1_weights, layer1_bias)))
        nn.add(DenseLayer(units=2, activation='sigmoid', weights=(layer2_weights, layer2_bias)))
        nn.predict(np.array([1]))

        self.assertEqual(str(layer1_weights), str(nn.layers[0].get_weights()[0]))
        self.assertEqual(str(layer1_bias), str(nn.layers[0].get_weights()[1]))

        self.assertEqual(str(layer2_weights), str(nn.layers[1].get_weights()[0]))
        self.assertEqual(str(layer2_bias), str(nn.layers[1].get_weights()[1]))

    def test_set_weights_setter(self):
        layer1_weights = np.array([
            [.5, .3]
        ])
        layer1_bias = np.array([-.1, -.6])

        layer2_weights = np.array([
            [.1, .4],
            [-.7, .2],
        ])
        layer2_bias = np.array([-.9, -.5])

        nn = NeuralNetwork()
        nn.add(DenseLayer(units=2, activation='sigmoid', input_shape=(1,)))
        nn.add(DenseLayer(units=2, activation='sigmoid'))

        nn.layers[0].set_weights((layer1_weights, layer1_bias))
        nn.layers[1].set_weights((layer2_weights, layer2_bias))

        nn.predict(np.array([1]))

        self.assertEqual(str(layer1_weights), str(nn.layers[0].get_weights()[0]))
        self.assertEqual(str(layer1_bias), str(nn.layers[0].get_weights()[1]))

        self.assertEqual(str(layer2_weights), str(nn.layers[1].get_weights()[0]))
        self.assertEqual(str(layer2_bias), str(nn.layers[1].get_weights()[1]))

    def test_saving_and_loading(self):
        nn = NeuralNetwork()
        nn.add(DenseLayer(units=2, activation='sigmoid', input_shape=(1,)))
        nn.add(DenseLayer(units=2, activation='sigmoid'))
        # do predict to initialize layers
        nn.predict(np.array([1]))
        nn.save('test_model.json')

        nn_from_file = NeuralNetwork.load('test_model.json')
        nn_from_file.predict(np.array([1]))

        for original_layer, layer_from_file in zip(nn.layers, nn_from_file.layers):
            self.assertEqual(str(original_layer.get_weights()), str(layer_from_file.get_weights()))


class RandomProbabilitySelectorTest(unittest.TestCase):

    def test1(self):
        elements = [
            ('name1', 0.05),
            ('name2', 0.1),
            ('name3', 0.15),
            ('name4', 0.4),
            ('name5', 0.3)
        ]
        elements_len = len(elements)
        result = {
            'name1': 0,
            'name2': 0,
            'name3': 0,
            'name4': 0,
            'name5': 0,
        }

        selected_elements_count = 2
        for i in range(50):
            selected_elements = random_probability_selection.perform_selection(elements, selected_elements_count)
            self.assertEqual(selected_elements_count, len(selected_elements))
            # check if the original list was not altered
            self.assertEqual(elements_len, len(elements))
            # check for repeated elements
            self.assertEqual(selected_elements_count, len(set(selected_elements)))
            for element in selected_elements:
                result[element] += 1

        print(result)


if __name__ == '__main__':
    unittest.main()
