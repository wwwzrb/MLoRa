import scipy.io as sio
import numpy as np
from scipy.fftpack import fft
from Stack import Stack, State
from lora_decode import LoRaDecode

class LoRa3:
    def __init__(self, bw, samples_per_second, sf, cr, powers, lengths):
        # Debugging
        self.debug_off = -1
        self.debug_info = 1
        self.debug_verbose = 2
        self.debug_on = self.debug_off
        self.d_debug = self.debug_on

        # Read all packets
        self.packets = None
        self.packets_shr = None
        self.d_powers = powers
        self.d_lengths = lengths
        self.packet_index = np.int(np.log2(lengths[0])) - 3
        self.payload_length = lengths[0]
        self.packet_length = np.int(lengths[0] + 3)
        self.packet_nums = len(self.d_powers)

        # Initialize signal info
        self.d_bw = bw
        self.d_samples_per_second = samples_per_second
        self.d_sf = sf
        self.d_cr = cr

        self.d_dt = 1.0 / self.d_samples_per_second
        self.d_decim_factor = int(self.d_samples_per_second // self.d_bw)
        self.d_number_of_bins = int(1 << self.d_sf)
        self.d_symbols_per_second = self.d_bw // self.d_number_of_bins
        self.d_samples_per_symbol = int(self.d_number_of_bins * self.d_decim_factor)
        self.d_half_number_of_bins = self.d_number_of_bins // 2
        self.d_half_samples_per_symbol = self.d_samples_per_symbol // 2
        self.d_quad_samples_per_symbol = self.d_samples_per_symbol // 4

        self.packet_header = [[124, 0, 0, 76, 4, 108, 8, 120],
                              [56, 0, 28, 124, 116, 96, 108, 40]]

        self.packet_chirp = \
            [[[124, 0, 0, 76, 4, 108, 8, 120,
               84, 71, 94, 85, 5, 122, 30, 123,
               58, 87, 49, 115, 106, 63, 24, 100,
               48, 68, 16, 123, 48, 71, 61, 87],

              [124, 0, 0, 76, 4, 108, 8, 120,
               1, 18, 67, 90, 66, 15, 31, 38,
               111, 2, 10, 108, 27, 42, 27, 33,
               50, 58, 80, 91, 32, 79, 57, 85],

              [124, 0, 0, 76, 4, 108, 8, 120,
               84, 71, 94, 85, 5, 122, 30, 123,
               58, 87, 49, 115, 106, 63, 24, 100,
               48, 68, 16, 123, 48, 71, 61, 87]],

             [[56, 0, 28, 124, 116, 96, 108, 40,
               84, 71, 94, 85, 5, 122, 30, 123,
               58, 87, 49, 115, 106, 63, 24, 100,
               63, 68, 8, 112, 73, 70, 99, 103,
               84, 32, 61, 69, 17, 116, 127, 118,
               60, 56, 45, 59, 47, 104, 33, 86],

              [56, 0, 28, 124, 116, 96, 108, 40,
               1, 18, 67, 90, 66, 15, 31, 38,
               111, 2, 10, 108, 27, 42, 27, 33,
               106, 17, 64, 80, 91, 109, 103, 18,
               1, 117, 83, 5, 53, 35, 119, 99,
               54, 66, 47, 69, 111, 64, 49, 92],

              [56, 0, 28, 124, 116, 96, 108, 40,
               84, 71, 94, 85, 5, 122, 30, 123,
               58, 87, 49, 115, 106, 63, 24, 100,
               63, 68, 8, 112, 73, 70, 99, 103,
               84, 32, 61, 69, 17, 116, 127, 118,
               60, 56, 45, 59, 47, 104, 33, 86]]]

        self.packet_byte = np.array(
            [[[8, 128, 144, 33, 67, 101, 135, 33, 67, 101, 135, 0, 0, 0, 0, 0, 0, 0, 0],
              [8, 128, 144, 18, 52, 86, 120, 18, 52, 86, 120, 0, 0, 0, 0, 0, 0, 0, 0],
              [8, 128, 144, 33, 67, 101, 135, 33, 67, 101, 135, 0, 0, 0, 0, 0, 0, 0, 0]],
             [[16, 129, 64, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135],
              [16, 129, 64, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120],
              [16, 129, 64, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135]]],
            dtype=np.uint8)

        # Build standard chirp
        self.d_downchirp = []
        self.d_downchirp_zero = []
        self.d_upchirp = []
        self.d_upchirp_zero = []

        # Read overlapped packet
        self.packet_idx = None
        self.chirp_cnt = None
        self.detect_scope = None
        self.chirp_offset = None
        self.samp_offset = None
        self.time_offset = None

        self.detect_offset_ab = None
        self.detect_offset_ac = None
        self.detect_chirp_offset_ac = None
        self.detect_chirp_offset_ab = None
        self.detect_samp_offset_ab = None
        self.detect_samp_offset_ac = None
        self.detect_bin_offset_ab = None
        self.detect_bin_offset_ac = None

        self.packet = None
        self.packet_shr = None
        self.packet_o = None

        # Initialize Decoding
        self.preamble_peak = 1
        self.bin_threshold = 9
        self.preamble_chirps = 6
        self.preamble_tolerance = 2
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_history = Stack()
        self.prebin_history = Stack()

        self.sfd_peak = 1
        self.sfd_chirps = 2
        self.sfd_tolerance = 4
        self.sfd_indexes = []
        self.sfd_bins = []
        self.sfd_begins = []
        self.sfd_history = []
        self.sfdbin_history = []
        self.sfdbins_history = []

        self.pld_peak = 1
        self.detect_offset = None
        self.detect_chirp_offset = None
        self.detect_samp_offset = None
        self.detect_bin_offset = None
        self.shift_samples = []
        self.pld_chirps = int(8 + (4 + self.d_cr) * np.ceil(self.d_lengths[0] * 2.0 / self.d_sf))
        self.pld_indexes = []
        self.pld_bins = []

        self.lora_decoder = LoRaDecode(self.payload_length, self.d_sf, self.d_cr)
        self.rec_indexes = []
        self.rec_bins = []
        self.corr_indexes = []
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_state = State.S_RESET

        # Save result
        self.chirp_errors_list = None
        self.decoded_prebins = []
        self.decoded_sfdbins = []
        self.decoded_chirps = []
        self.decoded_bytes = []
        self.decoded_chirp_errors = []
        self.decoded_header_errors = []
        self.decoded_byte_errors = []

    def read_packets(self):
        path_prefix = "../dbgresult/FFT05051921/"
        exp_prefix = ["DIST/", "PWR/", "SF/", "PWR_LEN/"]

        power = self.d_powers
        length = self.d_lengths
        sf = [self.d_sf, self.d_sf, self.d_sf]
        payload = ['inv', 'fwd', 'inv']

        exp_idx = 3

        file_name = []
        for idx in range(self.packet_nums):
            file_name.append(path_prefix + exp_prefix[exp_idx] + "data_" + payload[idx] + "_sf_" + str(sf[idx]) + "_len_" + str(length[idx]) + "_pwr_" + str(power[idx]) + ".mat")


        data = []
        for idx in range(self.packet_nums):
            data.append(sio.loadmat(file_name[idx]))

        self.packets = []
        self.packets_shr = []
        for idx in range(self.packet_nums):
            self.packets.append(data[idx]['packets'])
            self.packets_shr.append(data[idx]['packets_shr'])

    def build_ideal_chirp(self):
        T = -0.5 * self.d_bw * self.d_symbols_per_second
        f0 = self.d_bw / 2.0
        pre_dir = 2.0 * np.pi * 1j
        cmx = np.complex(1.0, 1.0)

        for idx in range(self.d_samples_per_symbol):
            t = self.d_dt * idx
            self.d_downchirp.append(np.exp(pre_dir * t * (f0 + T * t)))
            self.d_upchirp.append(np.exp(pre_dir * t * (f0 + T * t) * -1.0))

        self.d_downchirp_zero = np.concatenate(
            (self.d_downchirp[self.d_half_samples_per_symbol:], self.d_downchirp[0:self.d_half_samples_per_symbol]))
        self.d_upchirp_zero = np.concatenate(
            (self.d_upchirp[self.d_half_samples_per_symbol:], self.d_upchirp[0:self.d_half_samples_per_symbol]))

    def read_packet_shr(self, packet_idx, chirp_offset, samp_offset, packet_shr_o):
        # all time offset space
        # self.chirp_offset = np.random.randint(0, self.chirp_cnt)
        # self.samp_offset = np.random.randint(0, self.d_samples_per_symbol)
        # self.time_offset = self.chirp_offset * self.d_samples_per_symbol + self.samp_offset

        # controlled time offset space
        self.chirp_cnt = len(self.packets_shr[0][0]) // self.d_samples_per_symbol
        self.detect_scope = 2
        # self.chirp_offset = np.random.randint(0, self.chirp_cnt - 1)
        # self.samp_offset = self.d_decim_factor * np.random.randint(self.detect_scope + 1, self.d_number_of_bins - self.detect_scope)

        # configured time offset
        # self.chirp_offset = 0
        # self.samp_offset = 125 * self.d_decim_factor

        # get time offset from MLoRa2
        self.packet_idx = packet_idx
        self.chirp_offset = chirp_offset
        self.samp_offset = samp_offset
        self.time_offset = self.chirp_offset * self.d_samples_per_symbol + self.samp_offset

        self.packet_o = packet_shr_o

        if self.d_debug >= 1:
            self.show_info()

    def init_decoding(self):
        self.bin_threshold = 9
        self.preamble_chirps = 6
        self.preamble_tolerance = 2
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_history = Stack()
        self.prebin_history = Stack()

        self.sfd_chirps = 2
        self.sfd_tolerance = 4
        self.sfd_indexes = []
        self.sfd_bins = []
        self.sfd_begins = []
        self.sfd_history = []
        self.sfdbin_history = []
        self.sfdbins_history = []

        self.shift_samples = []
        self.pld_chirps = int(8 + (4 + self.d_cr) * np.ceil(self.d_lengths[0] * 2.0 / self.d_sf))
        self.pld_indexes = []
        self.pld_bins = []

        self.lora_decoder = LoRaDecode(self.payload_length, self.d_sf, self.d_cr)
        self.rec_indexes = []
        self.rec_bins = []
        self.corr_indexes = []
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_state = State.S_PREFILL

    def decoding_packet(self):
        idx = 0
        while idx<len(self.packet_o):
            if idx + self.d_samples_per_symbol > len(self.packet_o):
                break

            bgn_index = idx
            end_index = idx + self.d_samples_per_symbol
            chirp_o = self.packet_o[bgn_index:end_index]
            chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp_o, self.d_samples_per_symbol, self.detect_scope, self.preamble_peak)

            self.pre_history.push(chirp_index[0])
            self.prebin_history.push(chirp_max[0])
            if self.pre_history.size() > self.preamble_chirps:
                self.pre_history.pop_back()
                self.prebin_history.pop_back()

            if self.d_state == State.S_PREFILL:
                if self.pre_history.size() >= self.preamble_chirps:
                    self.d_state = State.S_DETECT_PREAMBLE
                else:
                    idx += self.d_samples_per_symbol
                    continue

            if self.d_state == State.S_DETECT_PREAMBLE:
                detect_index, detect_bin = self.detect_preamble(self.pre_history, self.prebin_history)
                if detect_bin[0] > 0:
                    self.d_state = State.S_SFD_SYNC
                    self.preamble_indexes.append(detect_index[0])
                    self.preamble_bins.append(detect_bin[0])

            if self.d_state == State.S_SFD_SYNC:
                bgn_sfd = idx - self.preamble_indexes[0] * self.d_decim_factor
                if not self.detect_preamble_chirp(0, idx, self.detect_scope, self.sfd_peak) \
                    and self.detect_down_chirp(bgn_sfd, self.detect_scope, self.sfd_peak):
                    self.d_state = State.S_READ_PAYLOAD
                    self.shift_samples.append(2*self.d_samples_per_symbol+self.d_quad_samples_per_symbol - self.preamble_indexes[0] * self.d_decim_factor)
                    self.sfd_begins.append(bgn_sfd)

            if self.d_state == State.S_READ_PAYLOAD:
                if len(self.pld_indexes) < self.pld_chirps:
                    chirp = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

                    bgn_index = idx + self.shift_samples[0]
                    end_index = bgn_index + self.d_samples_per_symbol
                    chirp[0:len(self.packet_o[bgn_index:end_index])] = self.packet_o[bgn_index:end_index]

                    chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                    self.pld_indexes.append(chirp_index[0])
                    self.pld_bins.append(chirp_max[0])

                if len(self.pld_indexes) >= self.pld_chirps:
                    self.d_state = State.S_STOP

            if self.d_state == State.S_STOP:
                self.recover_packet()
                self.error_correct()
                self.get_chirp_error()
                self.recover_byte()
                self.get_byte_error()

                break
            else:
                idx += self.d_samples_per_symbol
                continue

        self.save_result()

    def get_fft_bins(self, content, window, scope, num):
        curr_mul = np.multiply(content, self.d_downchirp[:window])
        curr_fft = np.abs(fft(curr_mul))

        # # only considering the previous half positive frequency bins and the last half negative frequency bins
        # # in total (0~d_decim_factor*d_bw)
        # curr_decim_fft = np.concatenate((curr_fft[:self.d_half_number_of_bins],
        #                                  curr_fft[(self.d_samples_per_symbol-self.d_half_number_of_bins):self.d_samples_per_symbol]))
        #
        # # positive frequency bins
        # curr_decim_fft = curr_fft[:self.d_number_of_bins]
        #
        # # negative frequency bins
        # curr_decim_fft = curr_fft[self.d_samples_per_symbol - self.d_number_of_bins:]

        # sum of pos and neg frequency bins
        curr_decim_fft = np.add(curr_fft[:self.d_number_of_bins],
                                curr_fft[self.d_samples_per_symbol - self.d_number_of_bins:])

        # curr_decim_fft = curr_fft

        curr_fft = curr_decim_fft
        # curr_max = np.amax(curr_fft)
        # curr_index = np.where((curr_max - curr_fft) < 0.1)[0]
        # curr_index = int(np.sum(curr_index) / len(curr_index))
        curr_index, curr_maxbin = self.get_max_bins(curr_fft, scope, num)

        return curr_index, curr_maxbin, curr_fft

    @staticmethod
    def get_max_bins(content, scope, num):
        curr_peaks = {}
        for idx in range(len(content)):
            peak_found = True
            for jdx in range(-scope, scope + 1):
                if content[idx] < content[(idx + jdx) % len(content)]:
                    peak_found = False
                    break
            if peak_found:
                curr_peaks[idx] = content[idx]
            # if(content[idx] > content[(idx-1) % len(content)]) and (content[idx] > content[(idx+1) % len(content)]):
            #     curr_peaks[idx] = content[idx]

        # the number of detected peaks is less than 3
        while len(curr_peaks) < 3:
            curr_peaks.update({-(len(curr_peaks) + 1): -(len(curr_peaks) + 1)})
        sorted_peaks = sorted(curr_peaks.items(), key=lambda kv: kv[1])
        max_idx = [sorted_peaks[-idx][0] for idx in range(1, num + 1)]
        max_bin = [sorted_peaks[-idx][1] for idx in range(1, num + 1)]
        # max_peaks = [(sorted_peaks[-idx][0], sorted_peaks[-idx][1]) for idx in range(3)]

        return max_idx, max_bin

    def detect_preamble(self, idx_stack, bin_stack):
        pre_idxes = [idx_stack.bottom()]
        pre_bins = [-1]

        pre_idx = idx_stack.bottom()
        pre_found = True

        for idx in range(idx_stack.size()):
            if abs(pre_idx - idx_stack.get_i(idx)) >= self.preamble_tolerance or bin_stack.get_i(idx) <= self.bin_threshold:
                pre_found = False
                break

        if pre_found:
            pre_idxes[0] = pre_idx
            pre_bins[0] = np.average(bin_stack.get_list()[1:])

        return pre_idxes, pre_bins

    def detect_preamble_chirp(self, pkt_idx, bgn_idx, scope, num):
        chirp_o = self.packet_o[bgn_idx:bgn_idx+self.d_samples_per_symbol]
        chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp_o, self.d_samples_per_symbol, scope, num)

        is_preamble_chirp = False

        if chirp_max[pkt_idx] > 0 and abs(chirp_index[pkt_idx] - self.preamble_indexes[pkt_idx]) < self.preamble_tolerance:
            is_preamble_chirp = True

        return is_preamble_chirp

    def detect_down_chirp(self, bgn_idx, scope, num):
        self.sfd_history.clear()
        self.sfdbin_history.clear()
        self.sfdbins_history.clear()

        for idx in range(self.sfd_chirps):
            pad_chirp = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            curr_chirp = self.packet_o[bgn_idx + idx * self.d_samples_per_symbol:bgn_idx + (idx + 1) * self.d_samples_per_symbol]
            pad_chirp[0:len(curr_chirp)] = curr_chirp
            sfd_idx, sfd_max, sfd_bin = self.get_down_chirp_bin(pad_chirp, scope, num)
            self.sfd_history.append(sfd_idx[0])
            self.sfdbin_history.append(sfd_max[0])
            self.sfdbins_history.append(sfd_bin)

        sfd_found = True
        for idx in range(self.sfd_chirps):
            curr_found = False
            if abs(self.sfd_history[idx] - self.d_half_number_of_bins) <= self.sfd_tolerance \
                    and self.sfdbin_history[idx] > self.bin_threshold:
                curr_found = True

            if not curr_found:
                sfd_found = False
                break

        if sfd_found:
            self.sfd_indexes.append(self.sfd_history)
            self.sfd_bins.append(np.sum(self.sfdbin_history))

            return True
        else:
            return False

    def get_down_chirp_bin(self, content, scope, num):
        curr_mul = np.multiply(content, self.d_upchirp_zero)
        curr_fft = np.abs(fft(curr_mul))

        # only considering the previous half positive frequency bins and the last half negative frequency bins
        # in total (0~d_decim_factor*d_bw)
        curr_decim_fft = np.concatenate((curr_fft[:self.d_half_number_of_bins],
                                         curr_fft[(self.d_samples_per_symbol - self.d_half_number_of_bins):self.d_samples_per_symbol]))

        curr_fft = curr_decim_fft
        curr_index, curr_maxbin = self.get_max_bins(curr_fft, scope, num)

        return curr_index, curr_maxbin, curr_fft

    def recover_packet(self):
        # copy demodulated result into rec_indexes
        self.rec_indexes = self.pld_indexes.copy()
        self.rec_bins = self.pld_bins.copy()

    def error_correct(self):
        index_error = 0
        for idx in range(8):
            ogn_index = self.rec_indexes[idx]
            curr_index = np.int16(np.rint(ogn_index/4))
            index_error += curr_index*4 - ogn_index
            self.corr_indexes.append(curr_index*4)

        index_error = np.rint(index_error/8)
        for idx in range(8, self.pld_chirps):
            ogn_index = self.rec_indexes[idx]
            curr_index = np.int16(np.mod(ogn_index+index_error, self.d_number_of_bins))
            self.corr_indexes.append(curr_index)

    def get_chirp_error(self):
        chirp_errors_list = np.abs(np.subtract(self.packet_chirp[self.packet_index][0], self.corr_indexes))
        self.chirp_errors.append(np.count_nonzero(chirp_errors_list))

    def recover_byte(self):
        self.rec_bytes.append(self.lora_decoder.decode(self.corr_indexes.copy()))

    def get_byte_error(self):
        comp_result = []
        header_result = []
        for idx in range(0, 3):
            header_result.append(bin(self.packet_byte[self.packet_index][0][idx] ^ self.rec_bytes[0][idx]).count('1'))
        for idx in range(0, self.packet_length):
            comp_result.append(bin(self.packet_byte[self.packet_index][0][idx] ^ self.rec_bytes[0][idx]).count('1'))

        self.byte_errors.append(np.sum(comp_result))
        self.header_errors.append(np.sum(header_result))

    def save_result(self):
        fail = True
        if len(self.sfd_begins) == 1:
            fail = False
            curr_prebins = self.preamble_bins.copy()
            curr_sfdbins = self.sfd_bins.copy()
            curr_chirps = self.corr_indexes.copy()
            curr_bytes = self.rec_bytes.copy()
            curr_chirp_errors = self.chirp_errors.copy()
            curr_header_errors = self.header_errors.copy()
            curr_byte_errors = self.byte_errors.copy()

            self.decoded_prebins.append(curr_prebins[0])
            self.decoded_sfdbins.append(curr_sfdbins[0])
            self.decoded_chirps.append(curr_chirps)
            self.decoded_bytes.append(curr_bytes)
            self.decoded_chirp_errors.append(curr_chirp_errors[0])
            self.decoded_header_errors.append(curr_header_errors[0])
            self.decoded_byte_errors.append(curr_byte_errors[0])

            if self.d_debug >= 2:
                self.show_result()

        if fail:
            if self.d_debug >=0:
                self.show_failed()

    def clear_result(self):
        # clear all saved results
        self.decoded_prebins.clear()
        self.decoded_sfdbins.clear()
        self.decoded_chirps.clear()
        self.decoded_bytes.clear()
        self.decoded_chirp_errors.clear()
        self.decoded_header_errors.clear()
        self.decoded_byte_errors.clear()


    def show_info(self):
        print("Show Info:")
        print("Packet Order: ", self.packet_idx)
        print("Chirp Offset: ", self.chirp_offset)
        print("Samp Offset: ", self.samp_offset)
        print("Time Offset: ", self.time_offset)
        print()

    def show_result(self):
        print("Show      result:")
        print("Chirp Offset: ", self.chirp_offset)
        print("Samp Offset: ", self.samp_offset)
        print("Time Offset: ", self.time_offset)

        print("Preamble  successful!")
        print("Preamble  indexes: ", self.preamble_indexes)
        print("Preamble  bins   : ", self.preamble_bins)
        print("SFD       indexes: ", self.sfd_indexes)
        print("SFD       bins   : ", self.sfd_bins)
        print("SFD       begins : ", self.sfd_begins)
        print("Downchirp successful!")
        print("PLD       indexes: ", self.pld_indexes)
        print("PLD       bins   : ", self.pld_bins)
        print("REC       indexes: ", self.rec_indexes)
        print("REC       bins   : ", self.rec_bins)
        print("CORR      indexes: ", self.corr_indexes)
        print("Packet     chirp : ", self.packet_chirp[self.packet_index][0])
        print("Chirp      Errors: ", self.chirp_errors)
        print("Header     Errors: ", self.header_errors)
        print("Byte       Errors: ", self.byte_errors)

    def show_failed(self):
        print("Sync      failed!")
        print("Chirp Offset: ", self.chirp_offset)
        print("Samp Offset: ", self.samp_offset)
        print("Time Offset: ", self.time_offset)
        print("Preamble  indexes: ", self.preamble_indexes)
        print("Preamble  bins   : ", self.preamble_bins)
        print("SFD       indexes: ", self.sfd_indexes)
        print("SFD       bins   : ", self.sfd_bins)
        print("SFD       begins : ", self.sfd_begins)

if __name__ == "__main__":
    test_bw = 125e3
    test_samples_per_second = 1e6
    test_sf = 7
    test_cr = 4
    test_powers = [0, -3]
    test_lengths = [8, 8]

    lora3 = LoRa3(test_bw, test_samples_per_second, test_sf, test_cr, test_powers, test_lengths)
    lora3.read_packets()
    lora3.build_ideal_chirp()

    for i in range(100):
        lora3.read_packet_shr(i, 0, 0, lora3.packets_shr[0][i])
        lora3.init_decoding()
        lora3.decoding_packet()
