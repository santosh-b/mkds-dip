from tensorflow import keras
import tensorflow as tf
import numpy as np

label_classes = {
    '(30, 32)': 0,
    '(25, 32)': 1,
    '(20, 32)': 2,
    '(15, 32)': 3,
    '(10, 32)': 4,
    '(05, 32)': 5,
    '(00, 00)': 6,
    '(05, 16)': 7,
    '(10, 16)': 8,
    '(15, 16)': 9,
    '(20, 16)': 10,
    '(25, 16)': 11,
    '(30, 16)': 12
}
inv_labels = {label_classes[x]: x for x in label_classes}


class Inference:

    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.viz = tf.keras.Model(inputs=self.model.inputs, outputs=[self.model.layers[i].output for i in [2,4,6,8,10]])

    def predict(self, img):
        x = self.model.predict(img[192:, :, :].reshape((1,192,256,3)))
        print(np.max(x), np.argmax(x))
        return inv_labels[np.argmax(x)] if (np.max(x) > 0) else '(00, 00)'

    def viz(self, img):
        return self.viz.predict(img)