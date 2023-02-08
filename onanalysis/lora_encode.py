import numpy as np
import utility as util


class LoRaEncode:

    def __init__(self, packet, sf=7, cr=4):
        self.packet = packet
        self.sf = sf
        self.cr = 4

        self.packet_len = self.packet[0] + 3
        self.payload_len = self.packet[0]
        if self.payload_len % self.sf != 0:
            self.packet_len = np.uint8(self.sf * np.ceil(self.payload_len * 1.0 / self.sf) + 3)

        self.num_bytes = self.packet_len * 2 - 1
        self.encoded = np.zeros(self.num_bytes, dtype=np.uint8)
        self.num_symbols = np.uint8(8 + 2 * (4 + cr) * np.ceil(self.payload_len * 1.0 / sf))
        self.symbols = np.zeros(self.num_symbols, dtype=np.uint16)

        self.debug_log("packet_length: " + str(self.packet_len))
        self.debug_log("num_bytes: " + str(self.num_bytes))
        self.debug_log("num_symbols: " + str(self.num_symbols))

        self.print_vector(self.packet, "Packet  ", 8)

        # Hamming Coding
        self.packet_offset = 3
        self.encode_offset = 5
        self.encoded[0:self.packet_offset * 2] = \
            self.fec_encode(self.packet[0:self.packet_offset], False)
        self.encoded[5:self.encode_offset + 2 * self.payload_len] = \
            self.fec_encode(self.packet[self.packet_offset:], True)
        self.print_vector(self.encoded, "Encoded ", 8)

        # Whitening
        util.whiten(self.encoded, self.encode_offset)
        self.print_vector(self.encoded, "Whiten  ", 8)

        # Shuffling
        self.shuffle_pattern = [1, 2, 3, 5, 4, 0, 6, 7]
        self.shuffle(self.encoded, self.shuffle_pattern)
        self.print_vector(self.encoded, "Shuffle ", 8)

        # Interleaving
        self.symbols_done = 0
        self.interleave_offset = 0

        self.interleave_offset += self.interleave_block(self.symbols, self.encoded, self.sf, self.cr, True)
        self.symbols_done += 8
        while (self.interleave_offset + self.sf) <= self.num_bytes:
            self.interleave_offset += self.interleave_block(self.symbols, self.encoded, self.sf, self.cr, False)
            self.symbols_done += 4 + self.cr

    @staticmethod
    def extract_bit(byte, pos):
        """
        Extract a bit from a given byte using MS ordering.
        ie. B7 B6 B5 B4 B3 B2 B1 B0
        """
        return (byte >> pos) & 0x01

    def hamming_encode_nibble(self, data):
        """
        Encode a nibble using Hamming encoding.
        Nibble is provided in form 0b0000DDDD == 0 0 0 0 D0 D1 D2 D3
        Encoded byte is in form H0 H1 D0 H2 D1 D2 D3 P
        """
        # Get data bits
        d = np.array([0, 0, 0, 0], np.uint8)
        d[0] = self.extract_bit(data, 3)
        d[1] = self.extract_bit(data, 2)
        d[2] = self.extract_bit(data, 1)
        d[3] = self.extract_bit(data, 0)

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

    def fec_encode(self, data, swap):
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
            en_result = self.hamming_encode_nibble(n)
            en_list.append(en_result)

        return en_list

    @staticmethod
    def shuffle(data, pattern):
        for i in range(len(data)):
            result = 0

            for j in range(8):
                if data[i] & (1 << pattern[j]):
                    result |= 1 << j

            data[i] = np.uint8(result)

    def interleave_block(self, syms, data, phy_sf=7, phy_cr=4, reduced_rate=False):
        if reduced_rate:
            phy_sf -= 2

        # Determine symbols for this block
        for i in range(4 + phy_cr):
            for j in range(phy_sf - 1, -1, -1):
                syms[i + self.symbols_done] |= ((data[j + self.interleave_offset] >> i) & 0x01) << j

            to_rotate = phy_sf - i
            if to_rotate < 0:
                to_rotate += phy_sf

            syms[i + self.symbols_done] = np.uint16(util.rotl(syms[i + self.symbols_done], to_rotate, phy_sf))

        self.print_vector(syms[self.symbols_done:self.symbols_done + 4 + phy_cr], "Chirps  ", phy_sf)

        # Determine bins
        for i in range(4 + phy_cr):
            syms[i + self.symbols_done] = util.gray_decode(syms[i + self.symbols_done])
            if reduced_rate:
                syms[i + self.symbols_done] <<= 2

        self.print_value(syms[self.symbols_done:self.symbols_done + 4 + phy_cr], "Bins    ")

        return phy_sf

    @staticmethod
    def debug_log(log):
        print(log)

    @staticmethod
    def print_vector(data, opt, width):
        print(opt, end=': ')
        for d in data:
            print("{0:0{1}b}".format(d, width), end=', ')
        print()

    @staticmethod
    def print_value(data, opt):
        print(opt, end=': ')
        for d in data:
            print(d, end=', ')
        print()


if __name__ == "__main__":
    test_sf = 7
    test_cr = 4
    test_packet = np.array([0x04, 0x80, 0xf0, 0x12, 0x34, 0x56, 0x78], np.uint8)
    lora_encoder = LoRaEncode(test_packet, test_sf, test_cr)

