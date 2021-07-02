# Resource 1: https://www.tensorflow.org/tutorials/keras/save_and_load
# pip install pyyaml h5py  # Required to save models in HDF5 format

import inspect

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import Callback
from Models.Callbacks import Callbacks
from Optimizers.Optimizers import Optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.models import Sequential

class Models(object):

    _class_name:str = None

    def __init__(self) -> None:  
        try:
            self._class_name = __class__.__name__
            print("Init Models class")
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

    def BuildModel(self,
              input_shape:tuple,
              number_of_classes:int,
              convolutional_activation:str = "relu",
              kernel_convolutuion:tuple = (3, 3),
              kernel_pooling:tuple = (2, 2),
              drop_out:float = 0.5,
              classification_activation:str = "softmax") -> any:
        try:
            input_layer = keras.Input(shape=input_shape)
            print("Given input_shape:" + str(input_shape))
            print("Given input_layer_shape:" + str(input_layer.shape))

            model = keras.Sequential(
                [
                    input_layer,
                    layers.Conv2D(32, kernel_size = kernel_convolutuion, activation = convolutional_activation),
                    layers.MaxPooling2D(pool_size = kernel_pooling),
                    layers.Conv2D(64, kernel_size = kernel_convolutuion, activation = convolutional_activation),
                    layers.MaxPooling2D(pool_size = kernel_pooling),
                    layers.Conv2D(128, kernel_size = kernel_convolutuion, activation = convolutional_activation),
                    layers.MaxPooling2D(pool_size = kernel_pooling),
                    layers.Flatten(),
                    layers.Dropout(drop_out),
                    layers.Dense(number_of_classes, activation = classification_activation),
                ]
            )

            return model
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

    def TrainModel(self,
              model:Sequential,
              x_train,
              y_train,
              batch_size:int = 128,
              epochs:int = 15,
              loss_function:str = "categorical_crossentropy",
              optimizer:str = "adam",
              metrics:list = ["accuracy"],
              validation_split:float = 0.1,
              callbacks:list = ["checkpoint"],
              verbose:int = 0) -> any:
        try:
            if(verbose > 0): 
                print("Show input shapes")
                print("x_train: " + str(x_train.shape))
                print("y_train: " + str(y_train.shape))

            if(verbose > 0): print("Set chosen optimizer.")
            opts:Optimizers = Optimizers()
            used_opt:Optimizer = opts.get_optimizer(name=optimizer)

            if(verbose > 0): print("Compile deep neural model.")
            model.compile(  loss=loss_function, 
                            optimizer=used_opt, 
                            metrics=metrics)

            model.summary()

            plot_model(model, to_file='model_1.png', show_shapes=True)
            
            if(verbose > 0): print("Start deep neural model training.")
            if (callbacks != None):

                history = model.fit(x_train, 
                                    y_train, 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    validation_split=validation_split,
                                    callbacks=Callbacks().get_callbacks(names= callbacks))
            else:
                history = model.fit(x_train, 
                                    y_train, 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    validation_split=validation_split)


            if(verbose > 0): print("Returning model train history.")
            return history

        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

    def EvaluateModel(self,
                 model:Sequential,
                 x_test,
                 y_test,
                 verbose:int = 0) -> any:
        try:
            score = model.evaluate(x_test, y_test, verbose=verbose)

            if (verbose > 0):
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])
            return score
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

    #//TODO: Missing functionality in Predict
    def Predict(self) -> any:
        try:
            return None
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

    #//TODO: Missing functionality in Store
    def Store(self) -> any:
        try:
            return None
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

    #//TODO: Missing functionality in Load
    def Load(self) -> any:
        try:
            return None
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

if __name__ == "__main__":
    Models().Build()