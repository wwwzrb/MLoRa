import numpy as np
import utility as util
import hamming as fec


def print_vector(data, opt, width):
    print(opt, end=': ')
    for d in data:
        print("{0:0{1}b}".format(d, width), end=', ')
    print()


def print_value(data, opt):
    print(opt, end=': ')
    for d in data:
        print(d, end=', ')
    print()


def deinterleave(syms, data, phy_sf=7, phy_cr=4, reduced_rate=False):
    if reduced_rate:
        phy_sf -= 2

    deinterleaved = np.zeros(phy_sf, np.uint8)

    bits_per_word = 4 + phy_cr
    offset_start = phy_sf - 1

    for i in range(bits_per_word):
        if reduced_rate:
            syms[i + symbols_done] /= 4
        syms[i + symbols_done] = util.gray_encode(syms[i + symbols_done])

        word = np.uint8(util.rotl(syms[i + symbols_done], i, phy_sf))

        j = (1 << offset_start)
        x = offset_start
        while j:
            if word & j:
                deinterleaved[x] |= (1 << i)
            x -= 1
            j >>= 1

    data[interleave_offset:interleave_offset+phy_sf] = deinterleaved[:]

    # print_vector(syms[symbols_done:symbols_done+bits_per_word], "S", phy_sf)
    # print_vector(deinterleaved[:], "D", 8)

    return 4 + phy_cr


def deshuffle(data, pattern):
    for i in range(len(data)):
        result = 0

        for j in range(len(pattern)):
            if data[i] & (1 << pattern[j]):
                result |= (1 << j)

        data[i] = np.uint8(result)


def fec_decode(data, swap):
    nibbles = []
    for d in data:
        de_result = fec.hamming_decode_byte(d)
        nibbles.append(de_result[0])

    de_list = []
    for i in range(len(data)//2):
        if swap:
            de_list.append((nibbles[2*i + 1] << 4) | nibbles[2*i])
        else:
            de_list.append((nibbles[2*i] << 4) | nibbles[2*i + 1])

    return de_list


symbols = np.array(
    [124, 60, 96, 112, 24, 108, 48, 100, 1, 18, 67, 90, 66, 15, 31, 38, 8, 19, 25, 108, 68, 102, 109, 9], np.uint32)

sf = 7
cr = 4
length = 4
packet_len = length + 3
payload_len = length

# Explicit header is always 3 bytes
packet = np.zeros(packet_len, dtype=np.uint8)
num_bytes = np.uint8(sf * np.ceil(2*payload_len/sf) + 6)
encoded = np.zeros(num_bytes, dtype=np.uint8)
num_symbols = np.uint8(8 + (4+cr)*np.ceil(payload_len*2.0/sf))

fec.debug_log("packet_length: " + str(packet_len))
fec.debug_log("num_bytes: " + str(num_bytes))
fec.debug_log("num_symbols: " + str(num_symbols))

print_vector(symbols, "Symbols ", 8)

# Deinterleave
symbols_done = 0
interleave_offset = 0

deinterleave(symbols, encoded, sf, cr, True)
symbols_done += 8
interleave_offset += 6
while (symbols_done + 4 + cr) <= num_symbols:
    symbols_done += deinterleave(symbols, encoded, sf, cr, False)
    interleave_offset += sf

print_vector(encoded, "After DI", 4 + cr)

# Deshuffle
shuffle_pattern = [5, 0, 1, 2, 4, 3, 6, 7]
deshuffle(encoded, shuffle_pattern)

print_vector(encoded, "After DS", 4 + cr)

# Dewhiten
encoded_offset = 6
util.whiten(encoded, encoded_offset)

print_vector(encoded, "After DW", 4 + cr)

# Hamming decoding
packet_offset = 3

packet[0:3] = fec_decode(encoded[0:encoded_offset], False)
packet[3:packet_len] = fec_decode(encoded[encoded_offset:encoded_offset+2*length], True)

print_vector(packet[0:length + 3], "After DH", 4 + cr)

