from base.base_loader import DataLoader
import tensorflow as tf
import tensorflow_datasets as tfds


class LoadData(DataLoader):
    def __init__(self, config):
        super().__init__(config)

    def load_data(self):
        self.dataset = {}
        print("Data loading started...")
        [self.train_data, self.test_data], self.info = tfds.load(name=self.config.data.name,
                                                                 split=["train", "test"],
                                                                 data_dir=self.config.data.data_dir,
                                                                 with_info=self.config.data.with_info)
        print("Loading finished!")
        assert isinstance(self.train_data, tf.data.Dataset)
        assert isinstance(self.test_data, tf.data.Dataset)

        self.dataset["num_classes"] = self.info.features["label"].num_classes
        self.dataset["input_shape"] = self.info.features["image"].shape
        self.dataset["train_num_samples"] = self.info.splits['train'].num_examples
        self.dataset["test_num_samples"] = self.info.splits['test'].num_examples
        self.image_size = self.config.data.image_size
        self.train_data = self.train_data.map(lambda image: self._preprocess_data(image,
                                                                                  self.dataset["num_classes"],
                                                                                  self.image_size),
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset["train"] = self.train_data.batch(
            self.config.data.batch_size)
        self.test_data = self.test_data.map(lambda image: self._preprocess_data(image,
                                                                                self.dataset["num_classes"],
                                                                                self.image_size),
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset["test"] = self.test_data.batch(
            self.config.data.batch_size)
        return self.dataset

    def _preprocess_data(self, datapoint, num_classes, image_size):
        input_image = datapoint["image"]
        label = datapoint["label"]
        input_image = tf.image.resize(input_image, (image_size, image_size))

        input_image = tf.cast(input_image, tf.float32) / 255.0
        onehot_label = tf.one_hot(label, depth=num_classes)

        return input_image, onehot_label
