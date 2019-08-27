.. _initial_setup:

Initial Setup
=============

Below are the steps to install and setup the package.


#. Pull the package from the github repo via::


        git clone https://github.com/wirrell/FAW


#. Make a virtual environment to install the requirements 
   (docs at https://docs.python.org/3/library/venv.html)::


        python3 -m venv faw_env


#. Activate the virtual environment via::

   
        source faw_env/bin/activate


#. Now navigate into the module and install the requirements using :code:`pip` via::


        cd FAW
        pip install -r requirements.txt


#. Next, add the module to the pythonpath. This must be done so that the module can be imported and import itself via the standard :code:`import {package}` syntax used in Python. I recommend doing this by going into the virtual environment directory at :code:`faw_env/lib64/python3.6/site-packages` and add a file name faw.pth which contains a single line that is the path to the package e.g.::


   /home/user/Documents/FAW


#. Now you can edit the :code:`config.json` file and modify the :code:`training_dir`, :code:`validation_dir`, and :code:`working_model` (where the model to be loaded by the classifier will be stored).

#. You can now edit the other parameters and run the :code:`train_model.py` script in the training dir to train a model before moving it in to your working model directory. The FAW package will now load the trained model when it is called upon to run a classification.
