Config = {
    "data": {
        "name": "mnist",
        "with_info": True,
        "data_dir": "./dataset",
        "image_size": 28,
        "batch_size": 32,
        "num_classes": 10,
        "validation_split": 0.1
    },
    "model": {
        "l1_filters": 32,
        "l2_filters": 64,
        "l3_filters": 128,
        "kernel_size": (3, 3),
        "l1_conv_activation": "linear",
        "l2_conv_activation": "linear",
        "l3_conv_activation": "linear",
        "padding": "same",
        "strides": 1,
        "leaky_alpha": 0.1,
        "pool_size": (2, 2),
        "dropout": "true",
        "dropout_l1": 0.2,
        "dropout_l2": 0.3,
        "dropout_l3": 0.3,
        "dropout_l4": 0.4,
        "l4_no_of_filters": 128,
        "dense_activation1": "linear",
        "dense_activation2": "softmax",
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "batch_size": 32,
        "epochs": 5,
        "checkpoint_path": "./model/checkpoints/cp.ckpt",
        "save_model_path": "./saved_model/baseline_model.h5"
    },
    "prune": {
        "pruning_epochs": 5,
        "initial_sparsity": 0.4,
        "final_sparsity": 0.7,
        "begin_step": 0,
        "save_prune_model_path": "./saved_model/pruned_model.h5",
        "save_quantized_model_path": "./saved_model/pruned_and_quantized_model.tflite"
    }
}
