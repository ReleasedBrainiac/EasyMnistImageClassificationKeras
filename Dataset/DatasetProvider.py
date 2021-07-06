import inspect
import numpy as np

from tensorflow import keras
from Support.SupportProvider import SupportProvider

class DatasetProvider(object):

    _class_name:str = None
    _support:SupportProvider = None

    def __init__(self) -> None:  
        try:

            print("Init DatasetProvider class")
            self._class_name = __class__.__name__
            self._support = SupportProvider()
            
        except Exception as ex:
            self._support.ExceptMessage(classname = self._class_name,
                                        funcname=inspect.currentframe().f_code.co_name,
                                        exception=ex)

    def LoadMnistNumberImages(self, verbose:int = 0) -> any:
        try:

            if (verbose > 0): 
                print("Load mnist number images dataset split in train and test set!")

            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

            if (verbose > 0): 
                print("Rescale images range to [0, 1]")

            x_train = x_train.astype("float32") / 255
            x_test = x_test.astype("float32") / 255

            if (verbose > 0): 
                print("Expand images with shape " + str(x_train.shape) + " to next higher dimension.")
                
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)

            shape_list = list(x_train.shape)
            shape_list.pop(0)
            input_shape = tuple(shape_list)

            if (verbose > 0): 
                print("x_train shape:", x_train.shape)
                print(x_train.shape[0], "train samples")
                print(x_test.shape[0], "test samples")

            return x_train, y_train, x_test, y_test, input_shape

        except Exception as ex:
            self._support.ExceptMessage(classname = self._class_name,
                                        funcname=inspect.currentframe().f_code.co_name,
                                        exception=ex)

    def ConvertClassesToBinaryVectors(self, 
                                      y_train:any,
                                      y_test:any,
                                      num_classes:int) -> any:
        try:

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            return y_train, y_test

        except Exception as ex:
            self._support.ExceptMessage(classname = self._class_name,
                                        funcname=inspect.currentframe().f_code.co_name,
                                        exception=ex)