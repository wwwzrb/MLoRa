import numpy as np

# List of syndrome positions. SYNDROME_CHECK[pos] will give the
# bit in the provided encoded byte that needs to be fixed
# Note: bit order used is 1 2 3 4 5 6 7 p
SYNDROME_CHECK = [-1, 7, 6, 5, 4, 3, 2, 1]


def debug_log(log):
    # print(log)
    debug = 0


def extract_bit(byte, pos):
    """
    Extract a bit from a given byte using MS ordering.
    ie. B7 B6 B5 B4 B3 B2 B1 B0
    """
    return (byte >> pos) & 0x01


def flip_bit(byte, pos):
    return byte ^ (1 << pos)


def hamming_encode_nibble(data):
    """
    Encode a nibble using Hamming encoding.
    Nibble is provided in form 0b0000DDDD == 0 0 0 0 D0 D1 D2 D3
    Encoded byte is in form H0 H1 D0 H2 D1 D2 D3 P
    """
    # Get data bits
    d = np.array([0, 0, 0, 0], np.uint8)
    d[0] = extract_bit(data, 3)
    d[1] = extract_bit(data, 2)
    d[2] = extract_bit(data, 1)
    d[3] = extract_bit(data, 0)

    # Calculate hamming bits
    h = np.array([0, 0, 0], np.uint8)
    h[0] = (d[0] + d[1] + d[3]) % 2
    h[1] = (d[0] + d[2] + d[3]) % 2
    h[2] = (d[1] + d[2] + d[3]) % 2

    # Calculate parity bit, using even parity
    p = 0 ^ d[0] ^ d[1] ^ d[2] ^ d[3] ^ h[0] ^ h[1] ^ h[2]

    # Encoded byte
    encoded = np.uint8(0)
    encoded |= np.uint8(
        (h[0] << 7) | (h[1] << 6) | (d[0] << 5) | (h[2] << 4) | (d[1] << 3) | (d[2] << 2) | (d[3] << 1) | p)

    return encoded


def hamming_decode_byte(byte):
    """
    Decode a single hamming encoded byte, and return a decoded nibble.
    Input is in form H0 H1 D0 H2 D1 D2 D3 P
    Decoded nibble is in form 0b0000DDDD == 0 0 0 0 D0 D1 D2 D3
    """
    error = 0
    correct = 0
    ogn = byte

    # Calculate syndrome
    s = np.array([0, 0, 0], np.uint8)

    # D0 + D1 + D3 + H0
    s[0] = (extract_bit(byte, 5) + extract_bit(byte, 3) + extract_bit(byte, 1) + extract_bit(byte, 7)) % 2

    # D0 + D2 + D3 + H1
    s[1] = (extract_bit(byte, 5) + extract_bit(byte, 2) + extract_bit(byte, 1) + extract_bit(byte, 6)) % 2

    # D1 + D2 + D3 + H2
    s[2] = (extract_bit(byte, 3) + extract_bit(byte, 2) + extract_bit(byte, 1) + extract_bit(byte, 4)) % 2

    syndrome = np.uint8((s[2] << 2) | (s[1] << 1) | s[0])

    # Check parity
    p = 0
    for i in range(0, 8):
        p ^= extract_bit(byte, i)

    if syndrome:
        if p:
            debug_log("One bit error which can be corrected")
            byte ^= (1 << SYNDROME_CHECK[syndrome])
            error = 1
            correct = 1
        else:
            error = 2
            correct = 0
            debug_log("Two bits error which can not be corrected")
    else:
        if p:
            error = 3
            correct = 0
            debug_log("Not sure! More than two or only parity")
        else:
            error = 0
            correct = 0
            debug_log("No bit error")

    debug_log("Origin  Result: " + "{0:8b}".format(ogn))
    debug_log("Correct Result: " + "{0:8b}".format(byte))

    # if syndrome:
    #     # Syndrome is not 0, so correct and log the error
    #     error += 1
    #     byte ^= (1 << SYNDROME_CHECK[syndrome])
    #     corrected += 1
    #
    # if p != extract_bit(byte, 0):
    #     # Parity bit is wrong, so log error
    #     if syndrome:
    #         # Parity is wrong and syndrome was also bad, so error is not corrected
    #         corrected -= 1
    #     else:
    #         # Parity is wrong and syndrome is fine, so corrected parity bit
    #         error += 1
    #         corrected += 1

    # Get data bits
    d = np.array([0, 0, 0, 0], np.uint8)
    d[0] = extract_bit(byte, 5)
    d[1] = extract_bit(byte, 3)
    d[2] = extract_bit(byte, 2)
    d[3] = extract_bit(byte, 1)

    decoded = np.uint8((d[0] << 3) | (d[1] << 2) | (d[2] << 1) | d[3])
    return decoded, error, correct


if __name__ == "__main__":
    test = np.uint8(0b1010)
    en_result = hamming_encode_nibble(test)
    print("Encode Result: "+"{0:8b}".format(en_result))

    for i in range(8):
        error_one = flip_bit(en_result, i)
        print(" ")
        de_result, de_error, de_correct = hamming_decode_byte(error_one)
        # print("Decode Result: " + "{0:b}".format(de_result))
        print("Decode Result: " + "{0:b}".format(de_result) + "; Error: ", de_error, "; Corr: ", de_correct)

    # # test for 0b1000
    # # one bit error
    # error_p = np.uint8(0b11100000)
    # error_s = np.uint8(0b01100001)
    # error_d = np.uint8(0b11000001)
    #
    # # two bit error
    # error_ps = np.uint8(0b01100000)
    # error_pd = np.uint8(0b11000000)
    # error_sd = np.uint8(0b01000001)
    # error_ss = np.uint8(0b00100001)
    # error_dd = np.uint8(0b11001001)
    # errors = [error_p, error_s, error_d, error_ps, error_pd, error_sd, error_ss, error_dd]
    #
    # for e in errors:
    #     print(" ")
    #     de_result, de_error, de_corrected = hamming_decode_byte(e)
    #     # print("Decode Result: " + "{0:b}".format(de_result))
    #     print("Decode Result: " + "{0:b}".format(de_result) + "; Error: ", de_error, "; Corr: ", de_corrected)

