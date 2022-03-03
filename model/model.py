from base.base_model import BaseModel
from utils.helpers import Helpers
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU


class Model(BaseModel):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def build_model(self):
        self.cnn_model = Sequential()

        self.cnn_model.add(Conv2D(filters=self.config.model.l1_filters,
                                  kernel_size=self.config.model.kernel_size,
                                  activation=self.config.model.l1_conv_activation,
                                  input_shape=self.dataset["input_shape"],
                                  padding=self.config.model.padding,
                                  strides=self.config.model.strides))
        self.cnn_model.add(LeakyReLU(alpha=self.config.model.leaky_alpha))
        self.cnn_model.add(MaxPooling2D(pool_size=self.config.model.pool_size))

        if self.config.model.dropout == "true":
            self.cnn_model.add(Dropout(self.config.model.dropout_l1))

        self.cnn_model.add(Conv2D(filters=self.config.model.l2_filters,
                                  kernel_size=self.config.model.kernel_size,
                                  activation=self.config.model.l2_conv_activation,
                                  padding=self.config.model.padding,
                                  strides=self.config.model.strides))
        self.cnn_model.add(LeakyReLU(alpha=self.config.model.leaky_alpha))
        self.cnn_model.add(MaxPooling2D(pool_size=self.config.model.pool_size))

        if self.config.model.dropout == "true":
            self.cnn_model.add(Dropout(self.config.model.dropout_l2))

        self.cnn_model.add(Conv2D(filters=self.config.model.l3_filters,
                                  kernel_size=self.config.model.kernel_size,
                                  activation=self.config.model.l3_conv_activation,
                                  padding=self.config.model.padding,
                                  strides=self.config.model.strides))
        self.cnn_model.add(LeakyReLU(alpha=self.config.model.leaky_alpha))
        self.cnn_model.add(MaxPooling2D(pool_size=self.config.model.pool_size))

        if self.config.model.dropout == "true":
            self.cnn_model.add(Dropout(self.config.model.dropout_l3))

        self.cnn_model.add(Flatten())

        self.cnn_model.add(Dense(units=self.config.model.l4_no_of_filters,
                                 activation=self.config.model.dense_activation1))

        self.cnn_model.add(LeakyReLU(alpha=self.config.model.leaky_alpha))
        if self.config.model.dropout == "true":
            self.cnn_model.add(Dropout(self.config.model.dropout_l4))

        self.cnn_model.add(Dense(units=self.config.data.num_classes,
                                 activation=self.config.model.dense_activation2))

        return self.cnn_model

    def compile_model(self):
        self.cnn_model.compile(optimizer=self.config.model.optimizer,
                               loss=self.config.model.loss,
                               metrics=self.config.model.metrics)
        return self.cnn_model

    def fit_model(self):
        return self.cnn_model.fit(self.dataset["train"],
                                  batch_size=self.config.model.batch_size,
                                  epochs=self.config.model.epochs,
                                  verbose=1)

    def evaluate_and_save_model(self):
        self.loss, self.accuracy = self.cnn_model.evaluate(
            self.dataset["test"])
        print("Test loss:", self.loss)
        print("Test accuracy:", self.accuracy)
        save_model(self.cnn_model, self.config.model.save_model_path,
                   include_optimizer=False)
        print("Saved baseline model to:", self.config.model.save_model_path)
        print("Size of baseline model: %.2f bytes" %
              (Helpers(self.config.model.save_model_path).get_gzipped_model()))
