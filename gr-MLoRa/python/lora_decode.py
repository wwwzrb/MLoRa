import numpy as np
import utility as util


class LoRaDecode:

    def __init__(self, length, sf=7, cr=4):
        self.debug = 0

        self.sf = sf
        self.cr = cr
        self.length = length
        self.packet_len = self.length + 3
        self.payload_len = self.length

        # Explicit header is always 3 bytes
        self.packet = np.zeros(self.packet_len, dtype=np.uint8)
        self.num_bytes = np.uint8(sf * np.ceil(2.0*self.payload_len/sf) + 6)
        self.encoded = np.zeros(self.num_bytes, dtype=np.uint8)
        self.num_symbols = np.uint8(8 + (4 + self.cr) * np.ceil(self.payload_len * 2.0 / self.sf))

        self.SYNDROME_CHECK = [-1, 7, 6, 5, 4, 3, 2, 1]

        self.debug_log("packet_length " + str(self.packet_len))
        self.debug_log("num_bytes " + str(self.num_bytes))
        self.debug_log("num_symbols " + str(self.num_symbols))

        self.symbols = None
        self.symbols_done = None

        self.interleave_offset = 0
        self.shuffle_pattern = [5, 0, 1, 2, 4, 3, 6, 7]
        self.encoded_offset = 6
        self.packet_offset = 3

    def decode(self, symbols):
        self.symbols = symbols
        self.print_vector(symbols, "Symbols ", 8)

        # De interleaving
        self.symbols_done = 0
        self.interleave_offset = 0

        self.deinterleave(self.symbols, self.encoded, self.sf, self.cr, True)
        self.symbols_done += 8
        self.interleave_offset += 6
        while (self.symbols_done + 4 + self.cr) <= self.num_symbols:
            self.symbols_done += self.deinterleave(self.symbols, self.encoded, self.sf, self.cr, False)
            self.interleave_offset += self.sf

        self.print_vector(self.encoded, "After DI", 4 + self.cr)

        # De shuffling
        self.deshuffle(self.encoded, self.shuffle_pattern)

        self.print_vector(self.encoded, "After DS", 4 + self.cr)

        # De whitening
        self.encoded_offset = 6
        util.whiten(self.encoded, self.encoded_offset)

        self.print_vector(self.encoded, "After DW", 4 + self.cr)

        # Hamming decoding
        self.packet_offset = 3

        self.packet[0:3] = self.fec_decode(self.encoded[0:self.encoded_offset], False)
        self.packet[3:self.packet_len] = self.fec_decode(
            self.encoded[self.encoded_offset:self.encoded_offset + 2 * self.payload_len], True)

        self.print_vector(self.packet[0:self.length + 3], "After DH", 4 + self.cr)

        self.print_value(self.packet[0:self.length + 3], "After DH")

        decoded_bytes = self.packet.copy()
        return decoded_bytes

    def deinterleave(self, syms, data, phy_sf=7, phy_cr=4, reduced_rate=False):
        if reduced_rate:
            phy_sf -= 2

        deinterleaved = np.zeros(phy_sf, np.uint8)

        bits_per_word = 4 + phy_cr
        offset_start = phy_sf - 1

        for i in range(bits_per_word):
            if reduced_rate:
                syms[i + self.symbols_done] = int(syms[i + self.symbols_done]/4)
            syms[i + self.symbols_done] = util.gray_encode(syms[i + self.symbols_done])

            word = np.uint8(util.rotl(syms[i + self.symbols_done], i, phy_sf))

            j = (1 << offset_start)
            x = offset_start
            while j:
                if word & j:
                    deinterleaved[x] |= (1 << i)
                x -= 1
                j >>= 1

        data[self.interleave_offset:self.interleave_offset + phy_sf] = deinterleaved[:]

        self.print_vector(syms[self.symbols_done:self.symbols_done+bits_per_word], "S", phy_sf)
        self.print_vector(deinterleaved[:], "D", 8)

        return 4 + phy_cr

    @staticmethod
    def deshuffle(data, pattern):
        for i in range(len(data)):
            result = 0

            for j in range(len(pattern)):
                if data[i] & (1 << pattern[j]):
                    result |= (1 << j)

            data[i] = np.uint8(result)

    @staticmethod
    def extract_bit(byte, pos):
        """
        Extract a bit from a given byte using MS ordering.
        ie. B7 B6 B5 B4 B3 B2 B1 B0
        """
        return (byte >> pos) & 0x01

    def hamming_decode_byte(self, byte):
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
        s[0] = (self.extract_bit(byte, 5) + self.extract_bit(byte, 3) + self.extract_bit(byte, 1) + self.extract_bit(byte, 7)) % 2

        # D0 + D2 + D3 + H1
        s[1] = (self.extract_bit(byte, 5) + self.extract_bit(byte, 2) + self.extract_bit(byte, 1) + self.extract_bit(byte, 6)) % 2

        # D1 + D2 + D3 + H2
        s[2] = (self.extract_bit(byte, 3) + self.extract_bit(byte, 2) + self.extract_bit(byte, 1) + self.extract_bit(byte, 4)) % 2

        syndrome = np.uint8((s[2] << 2) | (s[1] << 1) | s[0])

        # Check parity
        p = 0
        for i in range(0, 8):
            p ^= self.extract_bit(byte, i)

        if syndrome:
            if p:
                self.debug_log("One bit error which can be corrected")
                byte ^= (1 << self.SYNDROME_CHECK[syndrome])
                error = 1
                correct = 1
            else:
                error = 2
                correct = 0
                self.debug_log("Two bits error which can not be corrected")
        else:
            if p:
                error = 3
                correct = 0
                self.debug_log("Not sure! More than two or only parity")
            else:
                error = 0
                correct = 0
                self.debug_log("No bit error")

        self.debug_log("Origin  Result: " + "{0:8b}".format(ogn))
        self.debug_log("Correct Result: " + "{0:8b}".format(byte))

        # if syndrome:
        #     # Syndrome is not 0, so correct and log the error
        #     error += 1
        #     byte ^= (1 << self.SYNDROME_CHECK[syndrome])
        #     correct += 1
        #
        # if p != extract_bit(byte, 0):
        #     # Parity bit is wrong, so log error
        #     if syndrome:
        #         # Parity is wrong and syndrome was also bad, so error is not corrected
        #         correct -= 1
        #     else:
        #         # Parity is wrong and syndrome is fine, so corrected parity bit
        #         error += 1
        #         correct += 1

        # Get data bits
        d = np.array([0, 0, 0, 0], np.uint8)
        d[0] = self.extract_bit(byte, 5)
        d[1] = self.extract_bit(byte, 3)
        d[2] = self.extract_bit(byte, 2)
        d[3] = self.extract_bit(byte, 1)

        decoded = np.uint8((d[0] << 3) | (d[1] << 2) | (d[2] << 1) | d[3])
        return decoded, error, correct

    def fec_decode(self, data, swap):
        nibbles = []
        for d in data:
            de_result = self.hamming_decode_byte(d)
            nibbles.append(de_result[0])

        de_list = []
        for i in range(len(data) // 2):
            if swap:
                de_list.append((nibbles[2 * i + 1] << 4) | nibbles[2 * i])
            else:
                de_list.append((nibbles[2 * i] << 4) | nibbles[2 * i + 1])

        return de_list

    @staticmethod
    def debug_log(log):
        # print(log)
        debug = 0

    def print_vector(self, data, opt, width):
        if self.debug:
            # print(opt, end=': ')
            # for d in data:
            #     print("{0:0{1}b}".format(d, width), end=', ')
            print()

    def print_value(self, data, opt):
        if self.debug:
            # print(opt, end=': ')
            # for d in data:
            #     print(d, end=', ')
            print()


if __name__ == "__main__":
    test_sf = 7
    test_cr = 4
    test_length = 16
    test_symbols = np.array(
        [56,   0, 28, 124, 116,  96, 108, 40,
                        1, 18, 67, 90, 66, 15, 31, 38,
                        111, 2, 10, 108, 27, 42, 27, 33,
                        106, 17, 64, 80, 91, 109, 103, 18,
                        1, 117, 83, 5, 53, 35, 119, 99,
                        54, 66, 47, 69, 111, 64, 49, 92], np.uint16)
    lora_decoder = LoRaDecode(test_length, test_sf, test_cr)
    lora_decoder.decode(test_symbols)

