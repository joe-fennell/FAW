Deploying To AWS
================

Step-by-Step Guide

*NOTE:* A premium tier account will be needed to deploy to AWS. Free tier servers do not have sufficient memory to run the classification functions.

#. Login in to the AWS Management Console.
#. Choose the 'Launch a virtual machine' option in the 'Build a solution' section.
#. Select the Amazon Linux 2 AMI option.
#. Choose the instance type you required (a minimum of 2 vCPUs and 8 GB of memory is recommended)
#. Click Next: Configure Instance and keep all the defaults and click Next: Add Storage.
#. Select how much storage you think you will need. If you have modified the script to save image to a database, be sure to factor this in.
#. Click Next: Add Tags and then Next: Configure Security Group
#. Add a custom TCP rule with the port range set to the port you will declare in the module config.json file. (Make a note of the port for later.)
#. Set the CIDR, IP or Security Group field to 0.0.0.0/0
#. Click review and then finish. Be sure to set up a private key and download it. This will be needed to log in to the virtual machine.
#. Back at the management console, select the newly running instance and click connect.
#. Follow the instructions to connect and once connected, the initial linux packages you will need to install are: `git` and `python3`. This can be done via::

        sudo yum install git python3


#. Next install the OpenCV dependencies via::

        sudo yum install libXext libSM libXrender


#. Finally, install the package and setup the server via the :ref:`initial_setup` and :ref:`setting_server` tutorials.

**NOTE!:** Be sure to set the :code:`host_address` field to 0.0.0.0 for AWS and the port to the port you specified in Step 8.

