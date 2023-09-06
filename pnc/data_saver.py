import os

import numpy as np
import pickle


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DataSaver(metaclass=MetaSingleton):
    """
    Data Saver:
        add topics --> advance
    """
    def __init__(self, filename='pnc.pkl'):
        self._history = dict()
        if not os.path.exists('data'):
            os.makedirs('data')
        for f in os.listdir('data'):
            if f == filename:
                os.remove('data/' + f)
        self._file = open('data/' + filename, 'ab')

    def add(self, key, value):
        self._history[key] = value

    def advance(self):
        pickle.dump(self._history, self._file)

    def close(self):
        self._file.close()
