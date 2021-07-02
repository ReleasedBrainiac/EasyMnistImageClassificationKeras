"""
Short wrapper for keras > optimizers.py

Resources:

1. https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/

"""

import inspect
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, Adagrad, Adadelta, Adamax, Ftrl, SGD

class Optimizers(object):

    _class_name:str = None

    def __init__(self) -> None:  
        try:
            self._class_name = __class__.__name__
            print("Init Optimizers class")
        except Exception as ex:
            template = "An exception of type {exception} occurred in [{cname}.{fname}]. Arguments:\n{rest!r}"
            message = template.format(exception = type(ex).__name__, cname = self._class_name, fname = inspect.currentframe().f_code.co_name, rest = ex.args)
            print(message)

    #TODO: Proof functionality of all optimizers

    def get_optimizer(self, name:str,  clipvalue:float=20.0, learn_rate:float=0.001, amsgrad:bool=False, decay:float=0.004, epsilon:float=1e-07):
        try:
            if name.lower() == 'adadelta':
                return Adadelta(lr=learn_rate, rho=0.95, epsilon=epsilon, decay=decay, clipvalue=clipvalue)




            if name.lower() == 'adagrad':
                return Adagrad(lr=learn_rate, initial_accumulator_value=0.1, epsilon=epsilon, decay=decay, clipvalue=clipvalue)

            if name.lower() == 'adam':
                return Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay, amsgrad=amsgrad, clipvalue=clipvalue)

            if name.lower() == 'adamax':
                return Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=epsilon)

            if name.lower() == 'ftrl':
                return Ftrl(lr=learn_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, l2_shrinkage_regularization_strength=0.0, beta=0.0)

            if name.lower() == 'nadam':
                return Nadam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=epsilon, schedule_decay=decay, clipvalue=clipvalue)

            if name.lower() == 'rmsprop':
                return RMSprop(lr=learn_rate, clipvalue=clipvalue)

            if name.lower() == 'sgd':
                return SGD(lr=learn_rate, momentum=0.0, nesterov=False)

        except Exception as ex:
            template = "An exception of type {0} occurred in [ModelBuilder.get_optimizer]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message) 