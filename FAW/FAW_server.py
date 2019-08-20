"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

FAW_server.py

Script to coordinate remote server image reception and classification.

Speaks to a client and returns classification result via sockets.

NOTE: Multi-threaded server based on stackoverflow.com/questions/23828264/
"""

import socket
import threading
import logging
import argparse
import queue
import datetime
import numpy as np
import cv2
from FAW import FAW_classifier
from FAW import ImageCheck as IC
from FAW import ClassifierTools as CT

# TODO: add local database to store received images and metadata
config = CT.load_config()

HOST = config['server_settings']['host_address']
PORT = config['server_settings']['port']
# Client to Server communication codes
START = b'STRT'  # Beginning of message
GPS = b'GPSC'  # GPS delimiter (coords in form Lat, Long (decimal degrees)
LONG = b'LONG'  # Longitude delimiter
SOF = b'SOFT'  # Start of image file
END_MESSAGE = b'ENDM'
# Server to Client communication codes
DWLD_FAIL = b'FAIL'
DWLD_SUCC = b'SUCC'
INVALID = b'IVLD'
TRUE = b'TRUE'  # <-- Classification True, as in Fall Armyworm identified.
FALSE = b'FALS'
OBJECT_MISSING = b'MISS'
WORM_MISSING = b'NONE'
MULTIPLE_WORMS = b'MANY'
TOO_BLURRY = b'BLUR'

# Threshold above which CNN output is classified as positive result
CLASSIFICATION_THRESHOLD = config['classification_threshold']

# Set debugging options
parser = argparse.ArgumentParser()
parser.add_argument('-d', "--debug", action='store_true',
                    help="enable debugging")
args = parser.parse_args()
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

class FAWThreadedServer:
    """Threaded server to handle multiple client connections.

    NOTE: Uses Queues to schedule classification rather than running
          classification within individual threads because the
          classification_models module does not work if models are not loaded
          and run in the main thread.

    Example: Example message from client:
        STRTGPSC51.5138LONG-0.09847899999999754SOF{image byte string data}ENDM

    Args:
        host (str): Host address.
        port (int): Port to watch.

    Attributes:
        host (str): Host address.
        port (int): Port to watch.
        sock (socket.socket): Server socket listening for client connections.
        classifier (FAW_classifier.FAW_classifier): Fall Armyworm Classifier.
        classification_queue (queue.Queue): Queues the image to be classified.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.classifier = FAW_classifier.FAW_classifier()
        self.classification_queue = queue.Queue()
        logging.debug("Classifier loaded.")

    def start(self):
        """Start the server. Spawns thread to listen and starts classificatoin
        Queue loop in main thread."""
        threading.Thread(target=self.listen, daemon=True).start()
        self.monitor_loop()

    def listen(self):
        """Listen for client connections and start thread on connection."""
        self.sock.listen(5)
        logging.debug("Server started and listening.")
        while True:
            client, address = self.sock.accept()
            client.settimeout(30)
            threading.Thread(target=self.handle_client,
                             args=(client, address), daemon=True).start()

    def monitor_loop(self):
        """Monitors self.classification_queue for new images from client
        handling threads that need classification. Classifies new images and
        sends the result to the client before closing the connection."""
        while True:
            # Get the next image from queue. Will block until one is added.
            client, client_data = self.classification_queue.get()

            # process and classify the image then return the result to client
            result = self.process_image(client_data['img'])
            logging.debug("Conn {}: Image processing result: "
                          "{}".format(client_data['client_address'],
                                      result))

            if result == TRUE:
                client_data['FAW_classification'] = True
            if result == FALSE:
                client_data['FAW_classification'] = False

            try:
                # Send the result back to the client
                client.send(result)
                logging.debug("Conn {}: Result sent to client: "
                              "{}".format(client_data['client_address'],
                                          result))
                logging.debug("Conn {}: Closing connection to client".format(
                              client_data['client_address']))

                # Close the connection
                client.shutdown(socket.SHUT_RDWR)
                client.close()

            # TODO: add client data store method to store image and data in a
            # local database

            except socket.error as e:
                logging.debug("Conn {}: Socket error in monitor_loop: {}"
                              "".format(client_data['client_address'], e))

    def handle_client(self, client, address):
        """Interact with the client, check data and return results code.

        Args:
            client (socket.sockect): Socket that connects to the client.
            address: The address bound to the socket on the other end of the
            connection.
        """
        size = 4096  # byte size chunk to receive each pass
        logging.debug("Conn {}: Connected to client.".format(
            address))

        def respond_and_shutdown(response):
            """Send a response to the client and close the conneciton."""
            client.send(response)
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            logging.debug("Conn {}: Download failed. Sent code: {}".format(
                          address,
                          response))
            return

        try:
            first_signal = client.recv(4)  # Get first four bytes

            if first_signal != START:  # If starting delimiter not found
                respond_and_shutdown(DWLD_FAIL)
                return
            logging.debug("Conn {}: START_MESSAGE signal received.".format(
                address))

            message = b''
            while True:
                data = client.recv(size)

                if data[-4:] == END_MESSAGE:  # Delimiter for transfer end
                    message = message + data[:-4]
                    logging.debug("Conn {}: END_MESSAGE signal received."
                                  "".format(address))
                    break

                if data == b'':  # File transfer connection broke off
                    respond_and_shutdown(DWLD_FAIL)

                message = message + data

            #  after END_MESSAGE stop further receives from the client
            # to prevent double send
            client.shutdown(socket.SHUT_RD)

            # decode the message and extract the data
            is_valid, client_data = self.decode_message(message, address)

            if not is_valid:
                respond_and_shutdown(DWLD_FAIL)
                return

            client.send(DWLD_SUCC)
            logging.debug("Conn {}: Download successful. Sent code: {}".format(
                          address,
                          DWLD_SUCC))

            # queue the image for classification in the main thread
            self.classification_queue.put((client, client_data))

        except socket.error as e:
            logging.debug("Conn {}: Socket error in handle_client: {}".format(
                address, e))
            client.close()

    def decode_message(self, message, address):
        """Decode the message receieve from the client and sort data into dict.

        Args:
            client (socket.socket): client connected socket

        Returns:
            dict: containing client image and metadata.
        """
        client_data = {'received_datetime': str(datetime.datetime.now()),
                       'client_address': address,
                       'gps': None,
                       'img': b'',
                       'FAW_classification': None}

        if SOF not in message:
            return False, None

        [gps, img_data] = message.split(SOF)

        if gps[:4] != GPS or LONG not in gps:
            return False, None

        lat, lon = gps[4:].split(LONG)
        client_data['gps'] = (float(lat), float(lon))
        client_data['img'] = img_data

        return True, client_data

    def process_image(self,
                      image,
                      classification__threshold=CLASSIFICATION_THRESHOLD):
        """Processes the passed image and returns the result to be sent to the
        client.

        Args:
            image_stream (io.BytesIO): Byte stream sent from client.

        Returns:
            str: four letter result code to be sent back to the client.
        """
        valid, image = self.validate_and_load_image(image)
        if not valid:
            result = INVALID
            return result

        try:
            if self.classifier.predict(image) > classification__threshold:
                result = TRUE
            else:
                result = FALSE
        except IC.ObjectMissingError:
            result = OBJECT_MISSING
        except IC.WormMissingError:
            result = WORM_MISSING
        except IC.MultipleWormsError:
            result = MULTIPLE_WORMS
        except IC.TooBlurryError:
            result = TOO_BLURRY

        return result

    def validate_and_load_image(self, image_bytes):
        """Load the image from bytes and check it is valid.

        Args:
            image_bytes (bytes): byte string of image sent from client.

        Returns:
            bool: True if valid image, otherwise False.
            img (np.np_array): Loaded image if valid, otherwise None."""
        try:
            nparr = np.fromstring(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return False, None

        if img is None:
            return False, None

        return True, img


server = FAWThreadedServer(HOST, PORT)
server.start()
