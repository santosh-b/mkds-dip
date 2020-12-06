import socket
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle
import sys

HOST = '127.0.0.1'
PORT = 65432

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

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    model = keras.models.load_model('modelnew100.h5')
    #viz = tf.keras.Model(inputs=model.inputs, outputs=[model.layers[i].output for i in [2,4,6,8,10]])
    s.bind((HOST, PORT))
    s.listen()
    print('Server initialized on',(HOST,PORT))
    conn, addr = s.accept()
    with conn:
        print('Client Connection', addr)
        while True:
            data = conn.recv(294912)
            try:
                img = np.frombuffer(data, dtype=np.uint8).reshape(1,384,256,3)
            except:
                continue
            #inference = inv_labels[np.argmax(model.predict(img[:,:,:,:]))]
            #feature_maps = viz.predict(img[:,192:,:,:])
            #print(feature_maps[0].shape)
            conn.send(str.encode(str(model.predict(img[:,:,:,:]).tolist())))
            #conn.send(str.encode(str(viz.predict(img[:,:,:,:])[0].tolist())))
            #print(sys.getsizeof(feature_maps[0].tobytes()))
           # conn.send(feature_maps[0].tobytes())