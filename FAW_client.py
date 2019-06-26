"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

FAW_client.py

Client script to interact with FAW_server.py
"""

import socket
import pathlib

HOST = 'localhost'
PORT = 7777
# Client to Server communication codes
START = b'STRT'  # Beginning of message
GPS = b'GPSC'  # GPS delimiter (coords in form Lat, Long (decimal degrees))
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


def send_image_to_server(filepath, coords):
    """Send an image to FAW_server.py for classification.

    Args:
        filepath (str): path to the image file to be sent.
        coords (tuple): tuple of coords in (lattitude, longitude)

    Returns:
        valid (bool): True if image was valid
        result (bool or None): bool if valid image otherwise none. Bool is
                               classification result.
        error (str or None): Str if error message, otherwise None.
    """
    with open(path, 'rb') as f:
        img_bytes = f.read()

    sock.send(START)
    sock.send(GPS)
    sock.send(b'{}'.format(coords[0]))
    sock.send(LONG)
    sock.send(b'{}'.format(coords[1]))
    sock.send(SOF)
    sock.send(img_bytes)
    sock.send(END_MESSAGE)

    response_1 = sock.recv(4)
    response_2 = sock.recv(4)

    if response_1 != b'SUCC':
        valid, result, error = (False, None, "File transfer failed.")
        return valid, result, error

    if response_2 == b'TRUE':
        valid, result, error = (True, True, None)
    if response_2 == b'FALS':
        valid, result, error = (True, False, None)
    if response_2 == b'IVLD':
        valid, result, error = (False, None, "Not a valid image file.")
    if response_2 == b'MISS':
        valid, result, error = (False, None, "Not foreground object was "
                                "detected in the image.")
    if response_2 == b'NONE':
        valid, result, error = (False, None, "No caterpillar found in the image.")
    if response_2 == b'MANY':
        valid, result, error = (False, None, "More than one caterpillar found "
                                "in the image.")
    if response_2 == b'BLUR':
        valid, result, error = (False, None, "Image too blurry.")

    return valid, result, error








