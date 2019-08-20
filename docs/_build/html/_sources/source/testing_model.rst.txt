Testing A Trained Model
=======================

Once a model has been trained, it can be deployed to the FAW classifier by copying the contents of a training run save directory (e.g. all the files in say ``training/saves/0001/``) into the working model directory as specified in the ``config.json`` file.::

        { 
                ...
                "working_model": "/path/to/working/model/"
                ...
        }


It is recommended that by default you set this directory to be a folder separate from the ``/path/to/training/saves`` directory. I have used ``/explicit/path/to/FAW/model/``.

**NOTES:**

* Be sure to enter your file paths as absolute rather than relative file paths.
* Be sure to delete any existing files in the working model directory before copying a new model over or loading conflicts may occur.

Once the model has been copied over, any calls to the ``FAW`` module will load the new model. The ``FAW_classifier`` class will always load the model it finds in the working directory. This is true for both the classifier in isolation and when it is run as part of a server / AWS setup.
