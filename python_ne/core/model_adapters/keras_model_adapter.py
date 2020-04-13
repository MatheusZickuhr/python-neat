import numpy as np
from keras import Sequential
from keras.layers import Dense

from python_ne.core.model_adapters.model_adapter import ModelAdapter
from python_ne.core.model_adapters.keras_dense_layer_adapter import KerasDenseLayerAdapter


class KerasModelAdapter(ModelAdapter):
    def build_model(self):
        return Sequential()

    def add_dense_layer(self, **kwargs):
        self.model.add(Dense(**kwargs))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights()

    def predict(self, obs):
        return self.model.predict(obs)

    def get_layers(self):
        return [KerasDenseLayerAdapter(layer) for layer in self.model.layers]

    def save(self, file_path):
        self.model.save_weights(file_path)
