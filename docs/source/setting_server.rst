.. _setting_server:

Setting Up A Server
===================

Setting up a server involves running the ``FAW_server.py`` script.

This script takes its address and port settings from the below lines in the ``config.json`` file::

        {
                ...
                "server_settings": {
                        "host_address": 127.0.0.1,
                        "port": 7777
                }
        }

These can be set to anything the user requires.

To start the server with debugging message printing to the console, use the flag ``-d`` or ``--debug``

as in::

        >>python3 FAW_server.py -d

After the usual Tensorflow start up jargon, you will see the following messages if you are running in debug mode and the server has started properly::

        I0820 14:17:39.209626 140279839246144 FAW_server.py:82] Classifier loaded.
        I0820 14:17:39.210142 140276353455872 FAW_server.py:93] Server started and listening.

In debug mode, the server will post to console when it receives a message from a client program and when it responds. See the ``FAW_server.py`` and ``FAW_client.py`` docs for message formatting information.
