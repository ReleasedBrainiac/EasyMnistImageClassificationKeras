"""
Short wrapper for keras > callbacks.py

Resources:

1. https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
2. https://www.tensorflow.org/guide/keras/custom_callback

"""

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import Callback

class Callbacks(object):

    _epoch_border:int = 10

    def __init__(self) -> None:  
        try:
            print("Init " +__class__.__name__+ " class")
        except Exception as ex:
            raise

    def get_callbacks(self, 
                      names:list,
                      epochs:int = 20,
                      lr:float = 0.005):

        callbacks:list = []

        for name in names:
            name = name.lower()

            if (name == 'baselogger' or name == "base"): callbacks.append(self.BaseLogger())

            if (name == 'earlystopping' or name == "earlystop"): callbacks.append(self.EarlyStoppingOnMissingImprovement())

            if (name == 'reducelronplateau' or name == "reducelr"): callbacks.append(self.ReduceLearningrateOnPlateau())

            if (name == 'modelcheckpoint' or name == "checkpoint"): callbacks.append(self.ModelTrainStateCheckpoint())

            if (name == 'csvepochstream' or name == "csv"): callbacks.append(self.CSVEpochStreamLogger())

            if (name == 'recordhistory' or name == "history"): callbacks.append(self.RecordEventAsHistory())

            if (name == 'lrscheduler'): 
                scheduler = self.get_KerasExampleSchedulerFunc(epoch = epochs, lr = lr)
                callbacks.append(self.LearningRateScheduler(schedule_function=scheduler))

            if (name == 'metricslogger' or name == "metrics"): callbacks.append(self.MetricsToConsoleLogger())

            if (name == 'streamtoserver' or name == "server"): callbacks.append(self.StreamEventToServer())

            if (name == 'vizualizetensorboard' or name == "tensorboard"): callbacks.append(self.VisualizeOnTensorBoard())

            if (name == 'stoponnanloss' or name == "nanloss"): callbacks.append(self.StopTrainingOnNanLoss())

        if (len(callbacks) == 0):
            return None
        else:
            return callbacks

    def get_KerasExampleSchedulerFunc(self,
                                      epoch:int = 20,
                                      lr:float = 0.005):
        try:
            if epoch < self._epoch_border:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        except Exception as ex:
            raise


    def BaseLogger(self) -> Callback:
        try:
            return keras.callbacks.BaseLogger()
        except Exception as ex:
            raise

    def EarlyStoppingOnMissingImprovement(self,
                                          monitor_value:str = 'val_loss',
                                          monitor_mode:str = 'min',
                                          monitor_patience:int = 100) -> Callback:
        try:
            return keras.callbacks.EarlyStopping(monitor=monitor_value, 
                                                 mode=monitor_mode, 
                                                 patience=monitor_patience)
        except Exception as ex:
            raise

    def ReduceLearningrateOnPlateau(self,
                                    monitor_value:str = 'val_loss',
                                    factor:float = 0.02,
                                    patience:int = 5,
                                    min_learning_rate:float = 0.00005,
                                    verbose:int = 1) -> Callback:
        try:
            return keras.callbacks.ReduceLROnPlateau(monitor=monitor_value, 
                                                     factor=factor, 
                                                     patience=patience, 
                                                     min_lr=min_learning_rate, 
                                                     verbose=verbose)
        except Exception as ex:
            raise

    def ModelTrainStateCheckpoint(self, 
                                  checkpoint_path:str = './checkpoints/my_checkpoint',
                                  batch_size:int = 128, 
                                  verbose:int = 1,
                                  save_weights_only:bool = True,
                                  save_frequency_multiplier:int = 5,
                                  save_best_only:bool = True) -> Callback:
        try:
            return keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                   verbose=verbose, 
                                                   save_weights_only=save_weights_only,
                                                   save_freq=save_frequency_multiplier*batch_size,
                                                   save_best_only=save_best_only)
        except Exception as ex:
            raise

    def CSVEpochStreamLogger(self,
                             filename:str = "stream_log",
                             seperator:str = ',',
                             append_content:bool = False) -> Callback:
        try:
            return keras.callbacks.CSVLogger(filename=filename, separator=seperator, append=append_content)
        except Exception as ex:
            raise

    def RecordEventAsHistory(self) -> Callback:
        try:
            return keras.callbacks.History()
        except Exception as ex:
            raise

    def LearningRateScheduler(self,
                             schedule_function:any,
                             verbose:int = 1) -> Callback:
        try:
            return keras.callbacks.LearningRateScheduler(schedule=schedule_function, verbose=verbose)
        except Exception as ex:
            raise

    def MetricsToConsoleLogger(self,
                               count_mode:str = 'samples',
                               iterable_string_metrics:any = "accuracy") -> Callback:
        try:
            return keras.callbacks.ProgbarLogger(count_mode = count_mode, 
                                                 stateful_metrics = iterable_string_metrics)
        except Exception as ex:
            raise

    def StreamEventToServer(self,
                            server_root:str = 'http://localhost:9000',
                            dest_folder_path:str = '/publish/epoch/end/',
                            json_data:str = 'data',
                            http_headers:dict = None,
                            send_as_json:bool = False) -> Callback:
        try:
            return keras.callbacks.RemoteMonitor(root = server_root,
                                                 path = dest_folder_path,
                                                 field = json_data,
                                                 headers = http_headers,
                                                 send_as_json = send_as_json)
        except Exception as ex:
            raise

    def VisualizeOnTensorBoard(self,
                               logs_dest:str = 'logs',
                               epoch_activation_histogram_freq:int = 5,
                               visualize_graph:bool = True,
                               visualize_model_weights:bool = False,
                               update_freq:any = 'epoch', # 'epoch', 'batch' or an integer
                               profile_batch:int = 2,
                               visualize_embeddings_freq:int = 0,
                               embeddings_metadata = None,
                               **kwargs) -> Callback:
        try:
            return keras.callbacks.TensorBoard(log_dir = logs_dest,
                                               histogram_freq = epoch_activation_histogram_freq,
                                               write_graph = visualize_graph,
                                               write_images = visualize_model_weights,
                                               update_freq = update_freq,
                                               profile_batch = profile_batch,
                                               embeddings_freq = visualize_embeddings_freq,
                                               embeddings_metadata = embeddings_metadata,
                                               **kwargs)
        except Exception as ex:
            raise

    def StopTrainingOnNanLoss(self) -> Callback:
        try:
            return keras.callbacks.TerminateOnNaN()
        except Exception as ex:
            raise
