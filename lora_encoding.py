import numpy as np
import hamming as fec
import utility as util


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


def fec_encode(data, swap):
    nibbles = []
    for d in data:
        # Get and swap nibbles
        msb = np.uint8((d >> 4) & 0x0f)
        lsb = np.uint8(d & 0x0f)
        if swap:
            nibbles.append(lsb)
            nibbles.append(msb)
        else:
            nibbles.append(msb)
            nibbles.append(lsb)
    en_list = []
    for n in nibbles:
        en_result = fec.hamming_encode_nibble(n)
        en_list.append(en_result)

    return en_list


def shuffle(data, pattern):
    for i in range(len(data)):
        result = 0

        for j in range(8):
            if data[i] & (1 << pattern[j]):
                result |= 1 << j

        data[i] = np.uint8(result)


def interleave_block(syms, data, phy_sf=7, phy_cr=4, reduced_rate=False):
    if reduced_rate:
        phy_sf -= 2

    # Determine symbols for this block
    for i in range(4+phy_cr):
        for j in range(phy_sf-1, -1, -1):
            syms[i+symbols_done] |= ((data[j+interleave_offset] >> i) & 0x01) << j

        to_rotate = phy_sf - i
        if to_rotate < 0:
            to_rotate += phy_sf

        syms[i+symbols_done] = np.uint16(util.rotl(syms[i+symbols_done], to_rotate, phy_sf))

    print_vector(syms[symbols_done:symbols_done+4+phy_cr], "Chirps", phy_sf)

    # Determine bins
    for i in range(4+phy_cr):
        syms[i+symbols_done] = util.gray_decode(syms[i+symbols_done])
        if reduced_rate:
            syms[i+symbols_done] <<= 2

    print_value(syms[symbols_done:symbols_done+4+phy_cr], "Bins")

    return phy_sf


# packet = np.array([0x04, 0x80, 0xf0, 0x12, 0x34, 0x56, 0x78], np.uint8)
# packet = np.array([0x08, 0x80, 0x90, 0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78], np.uint8)
# packet = np.array([0x10, 0x81, 0x40,
#                    0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78,
#                    0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78], np.uint8)

packet = np.array([0x04, 0x80, 0xf0, 0x21, 0x43, 0x65, 0x87], np.uint8)
# packet = np.array([0x08, 0x80, 0x90, 0x21, 0x43, 0x65, 0x87, 0x21, 0x43, 0x65, 0x87], np.uint8)
# packet = np.array([0x10, 0x81, 0x40,
#                    0x21, 0x43, 0x65, 0x87, 0x21, 0x43, 0x65, 0x87,
#                    0x21, 0x43, 0x65, 0x87, 0x21, 0x43, 0x65, 0x87], np.uint8)

# packet = np.array([0x04, 0x80, 0xf0, 0x87, 0x65, 0x43, 0x21], np.uint8)
# packet = np.array([0x08, 0x80, 0x90, 0x87, 0x65, 0x43, 0x21, 0x87, 0x65, 0x43, 0x21], np.uint8)
# packet = np.array([0x10, 0x81, 0x40,
#                    0x87, 0x65, 0x43, 0x21, 0x87, 0x65, 0x43, 0x21,
#                    0x87, 0x65, 0x43, 0x21, 0x87, 0x65, 0x43, 0x21], np.uint8)

sf = 7
cr = 4
payload_len = packet[0]

packet_len = np.uint8(sf * np.ceil(2*payload_len/sf) + 6)

# delete last nibble to form 5*8 block
# compensate zero nibbles when (sf-2) > 6 to form (sf-2)*8 block
if sf == 7:
    num_bytes = packet_len - 1
else:
    num_bytes = packet_len
encoded = np.zeros(num_bytes, dtype=np.uint8)
num_symbols = np.uint8(8 + (4+cr)*np.ceil(payload_len*2.0/sf))
symbols = np.zeros(num_symbols, dtype=np.uint16)

fec.debug_log("packet_length: " + str(packet_len))
fec.debug_log("num_bytes: " + str(num_bytes))
fec.debug_log("num_symbols: " + str(num_symbols))

print_vector(packet, "Packet:  ", 8)

# Hamming coding
packet_offset = 3
encode_offset = 5
encoded[0:packet_offset*2] = fec_encode(packet[0:packet_offset], False)
encoded[encode_offset:encode_offset+2*payload_len] = fec_encode(packet[packet_offset:], True)
print_vector(encoded, "Encoded: ", 8)

# Whiten
util.whiten(encoded, encode_offset)
print_vector(encoded, "Whiten:  ", 8)

# Shuffle
shuffle_pattern = [1, 2, 3, 5, 4, 0, 6, 7]
shuffle(encoded, shuffle_pattern)
print_vector(encoded, "Shuffle: ", 8)

# Interleave
symbols_done = 0
interleave_offset = 0

interleave_offset += interleave_block(symbols, encoded, sf, cr, True)
symbols_done += 8
while (interleave_offset + sf) <= num_bytes:
    interleave_offset += interleave_block(symbols, encoded, sf, cr, False)
    symbols_done += 4 + cr
