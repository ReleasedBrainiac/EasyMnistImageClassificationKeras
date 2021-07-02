"""
Resources:

1. https://keras.io/
2. https://github.com/ReleasedBrainiac/GraphToSequenceNN
3. https://www.tensorflow.org/tutorials
4. https://keras.io/examples/vision/mnist_convnet/
5. https://www.tensorflow.org/tutorials/keras/classification
6. https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#when-you-re-puzzled-or-when-things-are-complicated
7. https://www.tensorflow.org/datasets
8. https://www.kaggle.com/tags/image-data
9. http://yann.lecun.com/exdb/mnist/index.html
10. https://www.tensorflow.org/guide/gpu

Necessary Libs and Frameworks (NOT nightly):

1. Tensorflow 2 (CPU/GPU) -> https://www.tensorflow.org/install
2. Graphviz -> https://graphviz.gitlab.io/download/
3. Pydot -> pip install pydot

"""



from Dataset.DatasetProvider import DatasetProvider
from Models.Models import Models
import inspect
import tensorflow as tf

class MnistDigitClassification():

    _class_name:str = None

    def __init__(self, verbose:int = 0) -> None:  
        try:
            gpus:int = len(tf.config.list_physical_devices('GPU'))
            print("Num GPUs Available: ", gpus)

            if (gpus > 0 and verbose > 0):
                tf.debugging.set_log_device_placement(True)

            self._class_name = __class__.__name__
            self.Execute()
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

    def Execute(self):

        try:

            print("Load and preprocess Mnist [0-9] dataset.")

            _num_classes_handwritten_digits:int = 10
            dspv:DatasetProvider = DatasetProvider()

            x_train, y_train, x_test, y_test, input_shape = dspv.LoadMnistNumberImages(verbose=1)
            y_train, y_test = dspv.ConvertClassesToBinaryVectors( y_test = y_test,
                                                                y_train = y_train,
                                                                num_classes = _num_classes_handwritten_digits)

            models:Models = Models()
            model = models.BuildModel(input_shape = input_shape,
                                      number_of_classes = _num_classes_handwritten_digits,
                                      convolutional_activation = "relu",
                                      kernel_convolutuion = (3, 3),
                                      kernel_pooling = (2, 2),
                                      drop_out = 0.5,
                                      classification_activation = "softmax")

            history = models.TrainModel(model = model,
                                        x_train = x_train,
                                        y_train = y_train,
                                        batch_size = 64,
                                        epochs = 20,
                                        loss_function = "categorical_crossentropy",
                                        optimizer = "adam",
                                        metrics = ["accuracy"],
                                        validation_split = 0.1,
                                        verbose = 1)

            score = models.EvaluateModel(model = model,
                                        x_test = x_test,
                                        y_test = y_test,
                                        verbose = 1)
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

if __name__ == "__main__":
    MnistDigitClassification(verbose = 0)