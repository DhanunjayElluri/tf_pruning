from model.model import Model
from base.base_pruning import BasePruning
from utils.helpers import Helpers
import os
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import save_model


class PruneModel(BasePruning):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.pruning_epochs = self.config.prune.pruning_epochs
        self.num_images = self.dataset["train_num_samples"]
        self.end_step = np.ceil(
            self.num_images / self.config.model.batch_size).astype(np.int32)

    def set_params(self):
        # Load functionality for adding pruning wrappers
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.config.prune.initial_sparsity,
                                                                     final_sparsity=self.config.prune.final_sparsity,
                                                                     begin_step=0,
                                                                     end_step=self.end_step)
        }

        # Loading pretrained model
        print("Loadint pretrained model....")
        if self.config.model.save_model_path is None:
            raise FileNotFoundError("Can't load model or Model does not exist")
        self.model = tf.keras.models.load_model(
            self.config.model.save_model_path)
        print("Model loaded successfully!")
        print("Pruning started....")
        self.model_for_prune = prune_low_magnitude(
            self.model, **pruning_params)
        return self.model_for_prune

    def compile_prune_model(self):
        return self.model_for_prune.compile(optimizer=self.config.model.optimizer,
                                            loss=self.config.model.loss,
                                            metrics=self.config.model.metrics)

    def fit_prune_model(self):
        self.callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        self.model_for_prune.fit(self.dataset["train"],
                                 batch_size=self.config.model.batch_size,
                                 epochs=self.pruning_epochs,
                                 verbose=1,
                                 callbacks=self.callbacks)
        return self.model_for_prune

    def evaluate_and_save_model(self):
        self.pruned_model_loss, self.pruned_model_accuracy = self.model_for_prune.evaluate(
            self.dataset["test"])
        print("Pruned model test loss:", self.pruned_model_loss)
        print("Pruned model test accuracy:", self.pruned_model_accuracy)
        # strip the pruning wrappers from the model
        self.model_for_export = tfmot.sparsity.keras.strip_pruning(
            self.model_for_prune)
        save_model(self.model_for_export,
                   self.config.prune.save_prune_model_path, include_optimizer=False)
        print("Saved pruned model to:",
              self.config.prune.save_prune_model_path)
        print("Size of pruned model: %.2f bytes" %
              (Helpers(self.config.prune.save_prune_model_path).get_gzipped_model()))

    def quantization(self):
        # Convert into TFLite model and convert with DEFAULT (dynamic range) quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(
            self.model_for_export)
        # dynamic range quantization which quantizes the weights, but not necessarily model activations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        # Save quantized model
        self.quantized_and_pruned_tflite_file = self.config.prune.save_quantized_model_path

        with open(self.quantized_and_pruned_tflite_file, 'wb') as f:
            f.write(tflite_model)
        print("Size of pruned and quantized model: %.2f bytes" %
              (Helpers(self.quantized_and_pruned_tflite_file).get_gzipped_model()))
        return self.quantized_and_pruned_tflite_file
