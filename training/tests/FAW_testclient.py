"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

FAW_testclient.py

Test client to test FAW_server.py

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

TEST_IMGS = str(pathlib.Path(__file__).parent) + '/imgs/'

# TODO: write full test cases with worms images in test folder to use in each
# of the test cases

# TODO: investigate train/114827 and validation/133615 blocking

# TODO: Add line of code to boot up server for testing so it doesn't have to be
# started manually by the user.

# TODO: finish image test cases

print("\n TESTING: \n\t FAW_testclient.py testing FAW_server.py.")
print("\n\t Start the server script in a separate process and then"
      " press any key to continue.")
input()

passed_count = 0
failed_count = 0


def message_test_case(message_list, expected_results, info_string):
    """Test that socket messages are handled correctly."""
    global passed_count, failed_count

    print("\n\nTEST: {}".format(info_string))
    print("\nMessage sequence: {}".format(message_list))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.connect((HOST, PORT))

    for message in message_list:
        sock.send(message)

    for expected in expected_results:
        response = sock.recv(4)  # standard length of each message
        if response != expected:
            print("\n\tResult: FAILED.")
            print("Expected server response {}, received {}.".format(
                expected, response))
            failed_count += 1
            return

    print("\n\tResult: PASSED.\n")
    passed_count += 1


def image_test_case(img, expected_results, info_string):
    """Test that images are handled correctly."""
    global passed_count, failed_count

    path = TEST_IMGS + img

    print("\n\nTEST: {}".format(info_string))
    print("\nTesting image handling of {}".format(path))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.connect((HOST, PORT))

    with open(path, 'rb') as f:
        img_bytes = f.read()

    sock.send(START)
    sock.send(GPS)
    sock.send(b'51.5138')
    sock.send(LONG)
    sock.send(b'-0.09847899999999754')
    sock.send(SOF)
    sock.send(img_bytes)
    sock.send(END_MESSAGE)

    response_1 = sock.recv(4)
    response_2 = sock.recv(4)
    responses = [response_1, response_2]

    for expected in expected_results:
        if expected not in responses:
            print("\n\tResult: FAILED.")
            print("Expected server response {}. Received {}.".format(
                expected_results, responses))
            failed_count += 1
            return

    print("\n\tResult: PASSED.\n")
    passed_count += 1


# Start message testing.
print("\n\n MESSAGE HANDLING TESTS.")

std_msg = [START, GPS, b'50.00', LONG, b'50.00', SOF, b'someimgbytes',
           END_MESSAGE]

# Start flag missing
message_test_case(std_msg[1:], [DWLD_FAIL, b''], "START flag missing.")
# GPS flag missing
message_test_case(std_msg[:1]+std_msg[2:], [DWLD_FAIL, b''], "GPS flag "
                  "missing.")
# LONG flag missing
message_test_case(std_msg[:3]+std_msg[4:], [DWLD_FAIL, b''], "LONG flag "
                  "missing.")
# SOF flag missing
message_test_case(std_msg[:5]+std_msg[6:], [DWLD_FAIL, b''], "SOF flag "
                  "missing.")
# END_MESSAGE flag missing
message_test_case(std_msg[:7], [b'', b''], "END_MESSAGE flag  missing. "
                  "(Timeout after 30 seconds expected.)")

# Start image handling testing

print("\n\n IMAGE HANDLING TESTS.")

# Empty image
image_test_case('empty.jpg', [DWLD_SUCC, OBJECT_MISSING], 'Empty image.')
# Not an image
image_test_case('not_an_img.txt', [DWLD_SUCC, INVALID], 'Not an image.')
# No worm present
image_test_case('not_a_worm.jpg', [DWLD_SUCC, WORM_MISSING], 'No worm in img.')
# Image too blurry
image_test_case('too_blurry.jpg', [DWLD_SUCC, TOO_BLURRY], 'Image too blurry.')
# Two worms
image_test_case('two_worms.jpg', [DWLD_SUCC, MULTIPLE_WORMS], 'Two worms in '
                'image.')

print("\n\n Finished. Total tests PASSED: {} FAILED: {}.".format(
    passed_count, failed_count))
