Training A Model
================

Model training is done in the ``training`` section of the package via the ``train_model.py`` script.

Training parameters are tuned by modifying the config.json file in the main directory of the package.

This file will be structured like so::

        { "training_dir": "/path/to/training/dir",
            "validation_dir": "/path/to/validation/dir",
            "batch_size": 1,
            "img_input_shape": [
                224,
                224,
                3
            ],
            "model_parameters": {
                "mlp_dropout_rate": 0.5,
                "mlp_optimizer_lr": 0.00001,
                "mlp_num_training_epochs": 1,
                "cnn_num_trainable_layers": 1,
                "cnn_learning_rate": 0.000001,
                "cnn_num_training_epochs": 1 
            },
            "working_model": "/path/to/working/model/for/deployment/",
            "classification_threshold": 0.5,
            "server_settings": {
                "host_address": "localhost",
                "port": 7777
            }
        }


**NOTE:** Be sure to enter your file paths as absolute rather than relative file paths.


The training script will use the two parameters ``training_dir`` and ``validation_dir`` to locate the images used for training. Remember to seperate the images in to seperate folders named after their respective classes i.e. ``faw`` and ``notfaw`` in this instance.

The field ``model_parameters`` is there to hold the model parameters for a given training run. This is what should be modified to vary the number of trainable layers, the learning rate, and the number of epochs etc.

**NOTE:** The model in use is based on ResNet50 from the Keras library. 

Upon starting the training script, you will be asked to enter a training reference number. This number will be added to all output files to keep them unique. The generated output files can be found in the ``training/saves/`` directory once training is complete. A typical file list will look like this::

        (1) 0001.log
        (2) 0001_bottleneck_fc_model.h5
        (3) 0001_bottleneck_history.json
        (4) 0001_cnn_history.json
        (5) 0001_cnn_model.json
        (6) 0001_cnn_weights.h5
        (7) 0001_config.json
        (8) 0001_train_valid_file_list.txt

#. The log file containing the console output for the training cycle.
#. The saved weights of the fully connected cap that is pre-trained before it is combined with the ResNet50 base.
#. The training and validation history from the training of the FC cap.
#. The training and validation history of the combined CNN model.
#. The saved structure of the combined CNN model.
#. The saved weights of the combined CNN model.
#. A copy of the config.json file in the main directory as it stood when the training run was started.
#. A list of the files used in training and validation for the training run.
