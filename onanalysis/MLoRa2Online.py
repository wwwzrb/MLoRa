import pickle as pl
import scipy
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from cmath import phase
from scipy.fftpack import fft
from Stack import State
from lora_decode import LoRaDecode
from LoRa2 import LoRa2
from OffResult import OffResult


class MLoRa2Online:

    def __init__(self, bw, samples_per_second, sf, cr, powers, byte_lens, chirp_lens):
        # Debugging
        self.debug_off = -1
        self.debug_info = 1
        self.debug_verbose = 2
        self.d_debug = self.debug_info

        # Read all packets
        self.packets_mlora2 = None
        self.d_powers = powers
        self.d_byte_lens = byte_lens
        self.d_chirp_lens = chirp_lens
        self.packet_index = np.int(np.log2(byte_lens[0])) - 3
        self.payload_length = byte_lens[0]
        self.payload_chirp_length = chirp_lens[0] - 12
        self.packet_length = np.int(byte_lens[0] + 3)
        self.packet_chirp_length = chirp_lens[0]

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

        self.packet_chirp = \
            [[[124, 0, 0, 76, 4, 108, 8, 120,
               1, 18, 67, 90, 66, 15, 31, 38,
               111, 2, 10, 108, 27, 42, 27, 33,
               50, 58, 80, 91, 32, 79, 57, 85],
              [124, 0, 0, 76, 4, 108, 8, 120,
               84, 71, 94, 85, 5, 122, 30, 123,
               58, 87, 49, 115, 106, 63, 24, 100,
               48, 68, 16, 123, 48, 71, 61, 87]],
             [[56, 0, 28, 124, 116, 96, 108, 40,
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

        self.packet_header = np.array([[124, 0, 0, 76, 4, 108, 8, 120], [56, 0, 28, 124, 116, 96, 108, 40]])

        self.packet_byte = np.array(
            [[[8, 128, 144, 18, 52, 86, 120, 18, 52, 86, 120, 0, 0, 0, 0, 0, 0, 0, 0],
              [8, 128, 144, 33, 67, 101, 135, 33, 67, 101, 135, 0, 0, 0, 0, 0, 0, 0, 0]],
             [[16, 129, 64, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120],
              [16, 129, 64, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135]]],
            dtype=np.uint8)

        # Build standard chirp
        self.d_downchirp = []
        self.d_downchirp_zero = []
        self.d_upchirp = []
        self.d_upchirp_zero = []

        # Read overlapped packet
        self.chirp_cnt = self.d_chirp_lens[0]
        self.packet_o = None

        # Initialize Decoding
        self.bgn_position = 0  # beginning position of the first packet
        self.end_position = None
        self.detect_scope = 2
        self.preamble_peak = 3
        self.bin_threshold = 9
        self.preamble_chirps = 6
        self.preamble_tolerance = 2
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_begins = []
        self.pre_idxes_list = []  # save all preamble_indexes
        self.pre_bins_list = []  # save all preamble_bins
        self.pre_bgns_list = []  # save all preamble_begins
        self.pre_history = []
        self.prebin_history = []
        for idx in range(3):
            self.pre_history.append([])
            self.prebin_history.append([])

        self.sfd_peak = 3
        self.sfd_threshold = 8
        self.sfd_chirps = 2
        self.sfd_tolerance = 4
        self.sfd_indexes = []
        self.sfd_bins = []
        self.sfd_cnts = [0, 0]
        self.sfd_idxes_list = []  # save all sfd_indexes
        self.sfd_bins_list = []  # save all sfd_bins
        self.sfd_cnts_list = []  # save all sfd_cnts
        self.sfd_history = []
        self.sfdbin_history = []
        self.sfdbins_history = []

        # Record SFD index of two packets
        self.consumed_0 = []
        self.consumed_1 = []
        self.sfd_history_0 = []
        self.sfdbin_history_0 = []
        self.sfd_history_1 = []
        self.sfdbin_history_1 = []

        self.pld_peak = 2
        self.detect_offset = None
        self.detect_chirp_offset = None
        self.detect_samp_offset = None
        self.detect_bin_offset = None
        self.shift_samples = []
        self.pld_chirps = int(8 + (4 + self.d_cr) * np.ceil(self.d_byte_lens[0] * 2.0 / self.d_sf))
        self.pld_tolerance = 2
        self.pld_indexes = [[], []]
        self.pld_bins = [[], []]

        self.lora_decoder = LoRaDecode(self.payload_length, self.d_sf, self.d_cr)
        self.rec_indexes = [[], []]
        self.rec_bins = [[], []]
        self.oly_indexes = []

        self.corr_indexes = []
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_states = [State.S_PREFILL, State.S_PREFILL]

        # Save result
        self.detect_offsets_list = []
        self.corr_idxes_list = []
        self.rec_bytes_list = []
        self.chirp_errors_list = []
        self.header_errors_list = []
        self.byte_errors_list = []

    def read_packets(self):
        path_prefix = '../data/mlora2_async'
        file_name = path_prefix

        self.packets_mlora2 = scipy.fromfile(file_name, scipy.complex64)
        # self.packet_o = self.packets_mlora2

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

    def init_decoding(self):
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_begins = []
        self.pre_history = []
        self.prebin_history = []
        for idx in range(3):
            self.pre_history.append([])
            self.prebin_history.append([])

        self.sfd_indexes = []
        self.sfd_bins = []
        self.sfd_cnts = [0, 0]

        # old implementation, not used now
        self.sfd_history = []
        self.sfdbin_history = []
        self.sfdbins_history = []

        # Record SFD index of two packets
        self.consumed_0 = []
        self.consumed_1 = []
        self.sfd_history_0 = []
        self.sfdbin_history_0 = []
        self.sfd_history_1 = []
        self.sfdbin_history_1 = []

        self.detect_offset = -1
        self.detect_chirp_offset = -1
        self.detect_samp_offset = -1
        self.shift_samples = []
        self.pld_indexes = [[], []]
        self.pld_bins = [[], []]
        self.rec_indexes = [[], []]
        self.rec_bins = [[], []]
        self.oly_indexes = []
        self.corr_indexes = []
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_states = [State.S_RESET, State.S_RESET]
        for idx in range(len(self.d_states)):
            if self.d_states[idx] == State.S_RESET:
                for jdx in range(3):
                    self.pre_history[jdx].clear()
                    self.prebin_history[jdx].clear()
                self.d_states[idx] = State.S_PREFILL

    def decoding_packet(self, content):
        self.packet_o = content

        idx = 0
        stop_cnt = 0
        num_consumed = self.d_samples_per_symbol

        if self.d_debug >= self.debug_verbose:
            print(len(self.packet_o))

        # Input length at least d_samples_per_symbol
        if len(self.packet_o) < self.d_samples_per_symbol:
            return 0

        # Always saving consecutive chips
        chirp_cnt = idx // self.d_samples_per_symbol
        bgn_index = idx
        end_index = idx + self.d_samples_per_symbol
        chirp_o = self.packet_o[bgn_index:end_index]
        chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp_o, self.d_samples_per_symbol, self.detect_scope,
                                                              self.preamble_peak)

        for jdx in range(self.preamble_peak):
            self.pre_history[jdx].append(chirp_index[jdx])
            self.prebin_history[jdx].append(chirp_max[jdx])

            if len(self.pre_history[jdx]) > self.preamble_chirps:
                self.pre_history[jdx].pop(0)
                self.prebin_history[jdx].pop(0)

        # Pre-filling chirps
        if self.d_states[0] == State.S_PREFILL and self.d_states[1] == State.S_PREFILL:
            if self.d_debug >= self.debug_verbose:
                print('State 0: ', self.d_states[0].name, 'State 1: ', self.d_states[1].name)

            if len(self.pre_history[0]) >= self.preamble_chirps:
                for jdx in range(2):
                    self.d_states[jdx] = State.S_DETECT_PREAMBLE

        # Preamble detection
        if self.d_states[0] == State.S_DETECT_PREAMBLE or self.d_states[1] == State.S_DETECT_PREAMBLE:
            detect_indexes, detect_bins = self.detect_preamble_all(self.pre_history, self.prebin_history)
            if self.d_states[0] == State.S_DETECT_PREAMBLE and self.d_states[1] == State.S_DETECT_PREAMBLE:

                if self.d_debug >= self.debug_verbose:
                    print('State 0: ', self.d_states[0].name, 'State 1: ', self.d_states[1].name)

                if len(np.where(detect_indexes >= 0)[0]) > 0:
                    repeat_index = np.where(detect_indexes >= 0)[0]
                    if len(repeat_index) >= 2:
                        for jdx in range(2):
                            self.d_states[jdx] = State.S_SFD_SYNC
                            self.preamble_indexes.append(detect_indexes[repeat_index[jdx]])
                            self.preamble_bins.append(detect_bins[repeat_index[jdx]])
                            self.pre_begins.append(chirp_cnt)

                            self.consumed_0 = self.packet_o[0:self.d_samples_per_symbol]
                            self.consumed_1 = self.packet_o[0:self.d_samples_per_symbol]

                        # align with the first packet
                        # i -= preamble_indexes[0] * d_decim_factor
                        # preamble_indexes[1] += preamble_indexes[0]
                    else:
                        self.d_states[0] = State.S_SFD_SYNC
                        self.preamble_indexes.append(detect_indexes[repeat_index[0]])
                        self.preamble_bins.append(detect_bins[repeat_index[0]])
                        self.pre_begins.append(chirp_cnt)

                        self.consumed_0 = self.packet_o[0:self.d_samples_per_symbol]

                        # align with the first packet
                        # i -= preamble_indexes[0] * d_decim_factor

            if self.d_states[0] != State.S_DETECT_PREAMBLE and self.d_states[1] == State.S_DETECT_PREAMBLE:

                if self.d_debug >= self.debug_verbose:
                    print('State 0: ', self.d_states[0].name, 'State 1: ', self.d_states[1].name)

                unique_index = np.where((detect_indexes >= 0) & (detect_indexes != self.preamble_indexes[0]))[0]
                repeat_index = np.where(detect_indexes >= 0)[0]
                if len(unique_index) > 0 and self.d_states[0] == State.S_SFD_SYNC and abs(
                        detect_indexes[unique_index[0]] - self.preamble_indexes[0]) >= self.preamble_tolerance:
                    self.d_states[1] = State.S_SFD_SYNC
                    self.preamble_indexes.append(detect_indexes[unique_index[0]])
                    self.preamble_bins.append(detect_bins[unique_index[0]])
                    self.pre_begins.append(chirp_cnt)

                    self.consumed_1 = self.packet_o[0:self.d_samples_per_symbol]

                if len(repeat_index) > 0 and self.d_states[0] != State.S_SFD_SYNC:
                    self.d_states[1] = State.S_SFD_SYNC
                    self.preamble_indexes.append(detect_indexes[repeat_index[0]])
                    self.preamble_bins.append(detect_bins[repeat_index[0]])
                    self.pre_begins.append(chirp_cnt)

                    self.consumed_1 = self.packet_o[0:self.d_samples_per_symbol]

        # Test preamble detection
        # if self.d_states[0] == State.S_SFD_SYNC and self.d_states[1] == State.S_SFD_SYNC:
        #     if self.d_debug >= self.debug_info:
        #         print('Preamble Success')
        #         print('Preamble Indexes: ', self.preamble_indexes)
        #         print('Preamble Bins   : ', self.preamble_bins)
        #         print('Preamble Begins : ', self.pre_begins)
        #
        #         self.pre_idxes_list.append(self.preamble_indexes)
        #         self.pre_bins_list.append(self.preamble_bins)
        #         self.pre_bgns_list.append(self.pre_begins)
        #
        #     self.init_decoding()

        if self.d_states[0] == State.S_SFD_SYNC or self.d_states[1] == State.S_SFD_SYNC:

            if self.d_states[0] == State.S_SFD_SYNC and self.d_states[1] != State.S_SFD_SYNC:
                if self.d_debug >= self.debug_verbose:
                    print('State 0: ', self.d_states[0].name, 'State 1: ', self.d_states[1].name)

                chirp_0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                # back part of previous chirp
                chirp_0[0:self.preamble_indexes[0] * self.d_decim_factor] = \
                    self.consumed_0[self.d_samples_per_symbol - self.preamble_indexes[0] * self.d_decim_factor:]
                # previous part of current chip
                chirp_0[self.preamble_indexes[0] * self.d_decim_factor:] = \
                    self.packet_o[0:self.d_samples_per_symbol - self.preamble_indexes[0] * self.d_decim_factor]

                # update history samples
                self.consumed_0 = self.packet_o[0:self.d_samples_per_symbol]

                # get downchirp index
                chirp_index_0, chirp_max_0, chirp_bin_0 = self.get_down_chirp_bin(chirp_0, self.detect_scope,
                                                                                  self.sfd_peak)

                self.sfd_history_0.append(chirp_index_0)
                self.sfdbin_history_0.append(chirp_max_0)

                if len(self.sfd_history_0) > 2:
                    self.sfd_history_0.pop(0)
                    self.sfdbin_history_0.pop(0)

                    if not self.detect_preamble_chirp(0, chirp_index, chirp_max) \
                            and self.detect_sfd_all(self.sfd_history_0, self.sfdbin_history_0):
                        self.d_states[0] = State.S_READ_PAYLOAD
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                                  - self.preamble_indexes[0] * self.d_decim_factor)

            if self.d_states[0] != State.S_SFD_SYNC and self.d_states[1] == State.S_SFD_SYNC:
                if self.d_debug >= self.debug_verbose:
                    print('State 0: ', self.d_states[0].name, 'State 1: ', self.d_states[1].name)

                chirp_1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                # back part of previous chirp
                chirp_1[0:self.preamble_indexes[1] * self.d_decim_factor] = \
                    self.consumed_1[self.d_samples_per_symbol - self.preamble_indexes[1] * self.d_decim_factor:]
                # previous part of current chip
                chirp_1[self.preamble_indexes[1] * self.d_decim_factor:] = \
                    self.packet_o[0:self.d_samples_per_symbol - self.preamble_indexes[1] * self.d_decim_factor]

                # update history samples
                self.consumed_1 = self.packet_o[0:self.d_samples_per_symbol]

                # get downchirp index
                chirp_index_1, chirp_max_1, chirp_bin_1 = self.get_down_chirp_bin(chirp_1, self.detect_scope,
                                                                                  self.sfd_peak)

                self.sfd_history_1.append(chirp_index_1)
                self.sfdbin_history_1.append(chirp_max_1)

                if len(self.sfd_history_1) > 2:
                    self.sfd_history_1.pop(0)
                    self.sfdbin_history_1.pop(0)

                    if not self.detect_preamble_chirp(1, chirp_index, chirp_max) \
                            and self.detect_sfd_all(self.sfd_history_1, self.sfdbin_history_1):
                        self.d_states[1] = State.S_READ_PAYLOAD
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                                  - self.preamble_indexes[1] * self.d_decim_factor)

            if self.d_states[0] == State.S_SFD_SYNC and self.d_states[1] == State.S_SFD_SYNC:
                if self.d_debug >= self.debug_verbose:
                    print('State 0: ', self.d_states[0].name, 'State 1: ', self.d_states[1].name)

                succ_cnt = 0

                chirp_0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                # back part of previous chirp
                chirp_0[0:self.preamble_indexes[0] * self.d_decim_factor] = \
                    self.consumed_0[self.d_samples_per_symbol - self.preamble_indexes[0] * self.d_decim_factor:]
                # previous part of current chip
                chirp_0[self.preamble_indexes[0] * self.d_decim_factor:] = \
                    self.packet_o[0:self.d_samples_per_symbol - self.preamble_indexes[0] * self.d_decim_factor]

                # update history samples
                self.consumed_0 = self.packet_o[0:self.d_samples_per_symbol]

                # get downchirp index
                chirp_index_0, chirp_max_0, chirp_bin_0 = self.get_down_chirp_bin(chirp_0, self.detect_scope,
                                                                                  self.sfd_peak)

                self.sfd_history_0.append(chirp_index_0)
                self.sfdbin_history_0.append(chirp_max_0)

                if len(self.sfd_history_0) > 2:
                    self.sfd_history_0.pop(0)
                    self.sfdbin_history_0.pop(0)

                    if not self.detect_preamble_chirp(0, chirp_index, chirp_max) \
                            and self.detect_sfd_all(self.sfd_history_0, self.sfdbin_history_0):
                        self.d_states[0] = State.S_READ_PAYLOAD
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                                  - self.preamble_indexes[0] * self.d_decim_factor)
                        succ_cnt += 1

                chirp_1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                # back part of previous chirp
                chirp_1[0:self.preamble_indexes[1] * self.d_decim_factor] = \
                    self.consumed_1[self.d_samples_per_symbol - self.preamble_indexes[1] * self.d_decim_factor:]
                # previous part of current chip
                chirp_1[self.preamble_indexes[1] * self.d_decim_factor:] = \
                    self.packet_o[0:self.d_samples_per_symbol - self.preamble_indexes[1] * self.d_decim_factor]

                # update history samples
                self.consumed_1 = self.packet_o[0:self.d_samples_per_symbol]

                # get downchirp index
                chirp_index_1, chirp_max_1, chirp_bin_1 = self.get_down_chirp_bin(chirp_1, self.detect_scope,
                                                                                  self.sfd_peak)

                self.sfd_history_1.append(chirp_index_1)
                self.sfdbin_history_1.append(chirp_max_1)

                if len(self.sfd_history_1) > 2:
                    self.sfd_history_1.pop(0)
                    self.sfdbin_history_1.pop(0)

                    if not self.detect_preamble_chirp(1, chirp_index, chirp_max) \
                            and self.detect_sfd_all(self.sfd_history_1, self.sfdbin_history_1):
                        self.d_states[1] = State.S_READ_PAYLOAD
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                                  - self.preamble_indexes[1] * self.d_decim_factor)
                        succ_cnt += 1

                if succ_cnt == 2:
                    # reverse preamble index
                    if self.preamble_indexes[0] < self.preamble_indexes[1]:
                        self.shift_samples.reverse()
                        self.preamble_indexes.reverse()

                # 同时检测到两个包的preamble, 只检测到一个数据包的SFD
                if succ_cnt == 1:
                    if self.d_states[0] == State.S_SFD_SYNC and self.d_states[1] == State.S_READ_PAYLOAD:
                        self.d_states[0] = State.S_READ_PAYLOAD
                        self.d_states[1] = State.S_SFD_SYNC
                        self.preamble_indexes.reverse()

        # Test packet synchronization
        # if self.d_states[0] == State.S_READ_PAYLOAD or self.d_states[1] == State.S_READ_PAYLOAD:
        #
        #     if self.d_states[0] == State.S_READ_PAYLOAD and self.d_states[1] != State.S_READ_PAYLOAD:
        #
        #         if self.d_debug >= self.debug_verbose:
        #             print('State 0: ', self.d_states[0].name, 'State 1: ', self.d_states[1].name)
        #         self.sfd_cnts[0] += 1
        #
        #     if self.d_states[0] != State.S_READ_PAYLOAD and self.d_states[1] == State.S_READ_PAYLOAD:
        #         self.sfd_cnts[0] += 1
        #         self.sfd_cnts[1] += 1
        #
        #     if self.d_states[0] == State.S_READ_PAYLOAD and self.d_states[1] == State.S_READ_PAYLOAD:
        #         self.sfd_cnts[0] += 1
        #         self.sfd_cnts[1] += 1
        #
        #         if self.d_debug >= self.debug_info:
        #             print('Preamble Success')
        #             print('Preamble Indexes: ', self.preamble_indexes)
        #             print('Preamble Bins   : ', self.preamble_bins)
        #             print('Preamble Begins : ', self.pre_begins)
        #
        #             self.pre_idxes_list.append(self.preamble_indexes)
        #             self.pre_bins_list.append(self.preamble_bins)
        #             self.pre_bgns_list.append(self.pre_begins)
        #
        #             print('SFD      Success')
        #             print('SFD      Indexes: ', self.sfd_indexes)
        #             print('SFD      Bins   : ', self.sfd_bins)
        #             print('SFD      Cnts : ', self.sfd_cnts)
        #
        #             self.sfd_idxes_list.append(self.sfd_indexes)
        #             self.sfd_bins_list.append(self.sfd_bins)
        #             self.sfd_cnts_list.append(self.sfd_cnts)
        #
        #         self.init_decoding()

        if self.d_states[0] == State.S_READ_PAYLOAD or self.d_states[1] == State.S_READ_PAYLOAD:

            if self.d_states[0] == State.S_READ_PAYLOAD and self.d_states[1] != State.S_READ_PAYLOAD:
                stop_cnt = 0
                self.sfd_cnts[0] += 1

                if len(self.pld_indexes[0]) < self.pld_chirps:
                    chirp_0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                    if self.preamble_indexes[0] * self.d_decim_factor >= self.d_quad_samples_per_symbol:
                        shift_chirps = 2

                        # consumed_samples in shift 2
                        consumed_samples = self.preamble_indexes[0] * self.d_decim_factor - self.d_quad_samples_per_symbol
                        chirp_0[0:consumed_samples] = self.consumed_0[self.d_samples_per_symbol - consumed_samples:]
                        chirp_0[consumed_samples:] = self.packet_o[0:self.d_samples_per_symbol - consumed_samples]

                    else:
                        shift_chirps = 3

                        # remain_samples in shift 3
                        remain_samples = self.d_quad_samples_per_symbol - self.preamble_indexes[0] * self.d_decim_factor
                        chirp_0[0:self.d_samples_per_symbol - remain_samples] = self.consumed_0[remain_samples:]
                        chirp_0[self.d_samples_per_symbol - remain_samples:] = self.packet_o[0:remain_samples]

                    if self.sfd_cnts[0] >= shift_chirps:
                        chirp_index_0, chirp_max_0, chirp_bin_0 = \
                            self.get_fft_bins(chirp_0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                        self.pld_indexes[0].append(chirp_index_0)
                        self.pld_bins[0].append(chirp_max_0)

                    self.consumed_0 = self.packet_o[0:self.d_samples_per_symbol]

                if len(self.pld_indexes[0]) >= self.pld_chirps:
                    self.d_states[0] = State.S_STOP

            if self.d_states[0] != State.S_READ_PAYLOAD and self.d_states[1] == State.S_READ_PAYLOAD:
                self.sfd_cnts[0] += 1
                self.sfd_cnts[1] += 1

                if len(self.pld_indexes[1]) < self.pld_chirps:
                    chirp_1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                    if self.preamble_indexes[1] * self.d_decim_factor >= self.d_quad_samples_per_symbol:
                        shift_chirps = 2

                        # consumed_samples in shift 2
                        consumed_samples = self.preamble_indexes[1] * self.d_decim_factor - self.d_quad_samples_per_symbol
                        chirp_1[0:consumed_samples] = self.consumed_1[self.d_samples_per_symbol - consumed_samples:]
                        chirp_1[consumed_samples:] = self.packet_o[0:self.d_samples_per_symbol - consumed_samples]

                    else:
                        shift_chirps = 3

                        # reamain_samples in shift 3
                        remain_samples = self.d_quad_samples_per_symbol - self.preamble_indexes[1] * self.d_decim_factor
                        chirp_1[0:self.d_samples_per_symbol - remain_samples] = self.consumed_1[remain_samples:]
                        chirp_1[self.d_samples_per_symbol - remain_samples:] = self.packet_o[0:remain_samples]

                    if self.sfd_cnts[1] >= shift_chirps:
                        chirp_index_1, chirp_max_1, chirp_bin_1 = \
                            self.get_fft_bins(chirp_1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                        self.pld_indexes[1].append(chirp_index_1)
                        self.pld_bins[1].append(chirp_max_1)

                    self.consumed_1 = self.packet_o[0:self.d_samples_per_symbol]

                if len(self.pld_indexes[1]) >= self.pld_chirps:
                    self.d_states[1] = State.S_STOP

            if self.d_states[0] == State.S_READ_PAYLOAD and self.d_states[1] == State.S_READ_PAYLOAD:
                stop_cnt = 0
                self.sfd_cnts[0] += 1
                self.sfd_cnts[1] += 1

                if self.preamble_indexes[0] > self.preamble_indexes[1]:
                    self.detect_bin_offset = self.preamble_indexes[0] - self.preamble_indexes[1]
                    self.detect_samp_offset = self.detect_bin_offset * self.d_decim_factor
                else:
                    self.detect_bin_offset = self.d_number_of_bins + self.preamble_indexes[0] - self.preamble_indexes[1]
                    self.detect_samp_offset = self.detect_bin_offset * self.d_decim_factor

                if len(self.pld_indexes[0]) < self.pld_chirps:
                    chirp_0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                    chirp_p0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                    chirp_b0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

                    if self.preamble_indexes[0] * self.d_decim_factor >= self.d_quad_samples_per_symbol:
                        shift_chirps = 2

                        # consumed_samples in shift 2
                        consumed_samples = self.preamble_indexes[0] * self.d_decim_factor - self.d_quad_samples_per_symbol
                        chirp_0[0:consumed_samples] = self.consumed_0[self.d_samples_per_symbol - consumed_samples:]
                        chirp_0[consumed_samples:] = self.packet_o[0:self.d_samples_per_symbol - consumed_samples]

                    else:
                        shift_chirps = 3

                        # reamain_samples in shift 3
                        remain_samples = self.d_quad_samples_per_symbol - self.preamble_indexes[0] * self.d_decim_factor
                        chirp_0[0:self.d_samples_per_symbol - remain_samples] = self.consumed_0[remain_samples:]
                        chirp_0[self.d_samples_per_symbol - remain_samples:] = self.packet_o[0:remain_samples]

                    chirp_p0[0:self.detect_samp_offset] = \
                        chirp_0[0:self.detect_samp_offset]
                    chirp_b0[self.detect_samp_offset:self.d_samples_per_symbol] = \
                        chirp_0[self.detect_samp_offset:self.d_samples_per_symbol]

                    if self.sfd_cnts[0] >= shift_chirps:
                        chirp_index_0, chirp_max_0, chirp_bin_0 = \
                            self.get_fft_bins(chirp_0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
                        chirp_index_p0, chirp_max_p0, chirp_bin_p0 = \
                            self.get_fft_bins(chirp_p0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
                        chirp_index_b0, chirp_max_b0, chirp_bin_b0 = \
                            self.get_fft_bins(chirp_b0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                        self.pld_indexes[0].append([chirp_index_0, chirp_index_p0, chirp_index_b0])
                        self.pld_bins[0].append([chirp_max_0, chirp_max_p0, chirp_max_b0])

                    self.consumed_0 = self.packet_o[0:self.d_samples_per_symbol]

                if len(self.pld_indexes[0]) >= self.pld_chirps:
                    self.d_states[0] = State.S_STOP

                if len(self.pld_indexes[1]) < self.pld_chirps:
                    chirp_1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                    chirp_p1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                    chirp_b1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                    if self.preamble_indexes[1] * self.d_decim_factor >= self.d_quad_samples_per_symbol:
                        shift_chirps = 2

                        # consumed_samples in shift 2
                        consumed_samples = self.preamble_indexes[1] * self.d_decim_factor - self.d_quad_samples_per_symbol
                        chirp_1[0:consumed_samples] = self.consumed_1[self.d_samples_per_symbol - consumed_samples:]
                        chirp_1[consumed_samples:] = self.packet_o[0:self.d_samples_per_symbol - consumed_samples]

                    else:
                        shift_chirps = 3

                        # reamain_samples in shift 3
                        remain_samples = self.d_quad_samples_per_symbol - self.preamble_indexes[1] * self.d_decim_factor
                        chirp_1[0:self.d_samples_per_symbol - remain_samples] = self.consumed_1[remain_samples:]
                        chirp_1[self.d_samples_per_symbol - remain_samples:] = self.packet_o[0:remain_samples]

                    chirp_p1[0:self.d_samples_per_symbol - self.detect_samp_offset] \
                        = chirp_1[0:self.d_samples_per_symbol - self.detect_samp_offset]
                    chirp_b1[self.d_samples_per_symbol - self.detect_samp_offset:self.d_samples_per_symbol] = \
                        chirp_1[self.d_samples_per_symbol - self.detect_samp_offset:self.d_samples_per_symbol]

                    if self.sfd_cnts[1] >= shift_chirps:
                        chirp_index_1, chirp_max_1, chirp_bin_1 = \
                            self.get_fft_bins(chirp_1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
                        chirp_index_p1, chirp_max_p1, chirp_bin_p1 = \
                            self.get_fft_bins(chirp_p1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
                        chirp_index_b1, chirp_max_b1, chirp_bin_b1 = \
                            self.get_fft_bins(chirp_b1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                        self.pld_indexes[1].append([chirp_index_1, chirp_index_p1, chirp_index_b1])
                        self.pld_bins[1].append([chirp_max_1, chirp_max_p1, chirp_max_b1])

                    self.consumed_1 = self.packet_o[0:self.d_samples_per_symbol]

                if len(self.pld_indexes[1]) >= self.pld_chirps:
                    self.d_states[1] = State.S_STOP

        if self.d_states[0] == State.S_STOP and self.d_states[1] == State.S_STOP:

            if self.d_states[0] == State.S_STOP and self.d_states[1] == State.S_DETECT_PREAMBLE:
                stop_cnt += 1

                if stop_cnt >= 6:
                    stop_cnt = 0
                    self.recover_single_packet()
                    self.error_correct()
                    self.get_chirp_error()
                    self.recover_byte()
                    self.get_byte_error()
                    self.show_result()
                    self.save_result()

                    self.bgn_position = (chirp_cnt + 1) * self.d_samples_per_symbol

                    self.init_decoding()

            if self.d_states[0] == State.S_STOP and self.d_states[1] == State.S_STOP:
                if self.preamble_indexes[0] >= self.preamble_indexes[1]:
                    self.detect_chirp_offset = self.sfd_cnts[0] - self.sfd_cnts[1]
                    self.detect_bin_offset = self.preamble_indexes[0] - self.preamble_indexes[1]
                    self.detect_samp_offset = self.detect_bin_offset * self.d_decim_factor
                else:
                    self.detect_chirp_offset = self.sfd_cnts[0] - self.sfd_cnts[1] - 1
                    self.detect_bin_offset = self.d_number_of_bins + self.preamble_indexes[0] - self.preamble_indexes[1]
                    self.detect_samp_offset = self.detect_bin_offset * self.d_decim_factor

                self.recover_packet()
                self.error_correct()
                self.get_chirp_error()
                self.recover_byte()
                self.get_byte_error()
                self.show_result()
                self.save_result()

                self.init_decoding()

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

    def detect_preamble_all(self, idx_stacks, bin_stacks):
        pre_idxes = np.array([-idx - 1 for idx in range(len(idx_stacks))])
        pre_bins = np.full(len(idx_stacks), -1, dtype=float)
        for idx in range(len(idx_stacks)):
            pre_idx = idx_stacks[idx][0]
            pre_found = True

            curr_idx = []
            curr_pos = []
            for jdx in range(self.preamble_chirps):
                curr_found = False
                for kdx in range(len(idx_stacks)):
                    if abs(pre_idx - idx_stacks[kdx][jdx]) < self.preamble_tolerance and \
                            bin_stacks[kdx][jdx] > self.bin_threshold:
                        curr_idx.append((jdx, kdx))
                        curr_pos.append(kdx)
                        curr_found = True
                        break
                if not curr_found:
                    pre_found = False
                    break
                # if not curr_found or (len(np.unique(curr_pos)) >= 3):
                #     pre_found = False
                #     break

            if pre_found:
                # save current preamble index
                pre_idxes[idx] = pre_idx
                # save max bin of preamble, the first chirp is excluded to avoid incompleteness
                pre_bins[idx] = np.average([bin_stacks[bin_idx[1]][bin_idx[0]] for bin_idx in curr_idx[1:]])

        return pre_idxes, pre_bins

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

    def detect_preamble_chirp(self, pkt_idx, chirp_idx, chirp_max):
        is_preamble_chirp = False

        for idx in range(len(chirp_idx)):
            if chirp_max[idx] > 0 and abs(chirp_idx[idx]-self.preamble_indexes[pkt_idx]) < self.preamble_tolerance:
                is_preamble_chirp = True
                break

        return is_preamble_chirp

    def detect_sfd_all(self, sfd_history, sfdbin_history):
        sfd_found = True

        curr_idx = []
        for idx in range(self.sfd_chirps):
            curr_found = False
            for jdx in range(3):
                if abs(sfd_history[idx][jdx] - self.d_half_number_of_bins) <= self.sfd_tolerance \
                        and sfdbin_history[idx][jdx] > self.sfd_threshold:
                    curr_idx.append((idx, jdx))
                    curr_found = True
                    break

            if not curr_found:
                sfd_found = False
                break

        # if bgn_idx == sfd_bgns[0] or bgn_idx == sfd_bgns[1]:
        #     for idx in range(sfd_chirps):
        #         bgn = bgn_idx + idx*d_samples_per_symbol
        #         end = bgn + d_samples_per_symbol
        #
        #         if bgn_idx == sfd_bgns[0]:
        #             show_signal([sfdbins_history[idx], np.zeros(d_number_of_bins)], 'sfd ' + str(idx))
        #             show_chirp(bgn, end, 0)
        #         else:
        #             show_signal([sfdbins_history[idx], np.zeros(d_number_of_bins)], 'sfd ' + str(idx))
        #             show_chirp(bgn, end, 1)

        if sfd_found:
            self.sfd_indexes.append([sfd_history[bin_idx[0]][bin_idx[1]] for bin_idx in curr_idx])
            self.sfd_bins.append(np.sum([sfdbin_history[bin_idx[0]][bin_idx[1]] for bin_idx in curr_idx]))

            return True
        else:
            return False

    def recover_single_packet(self):

        for idx in range(len(self.pld_indexes)):
            if len(self.pld_indexes[idx]) == self.pld_chirps:
                for jdx in range(self.pld_chirps):
                    self.rec_indexes[idx].append(self.pld_indexes[idx][jdx][np.argmax(self.pld_bins[idx][jdx])])
                    self.rec_bins[idx].append(np.max(self.pld_bins[idx][jdx]))

    def recover_packet(self):
        # preamble chirp index of B when aligning with chirp of A
        preamble_shift = (self.d_quad_samples_per_symbol - self.preamble_indexes[
            0] * self.d_decim_factor) // self.d_decim_factor
        preamble_index = (self.preamble_indexes[1] + preamble_shift + self.d_number_of_bins) % self.d_number_of_bins

        samp_offset_a = {1: self.detect_samp_offset, 2: self.d_samples_per_symbol - self.detect_samp_offset}
        samp_offset_a = np.array(sorted(samp_offset_a.items(), key=lambda x: x[1]))
        ref_seg_a = samp_offset_a[-1][0]

        # preamble of A is collision-free
        for idx in range(0, max(self.detect_chirp_offset - 12, 0)):
            if not (type(self.pld_indexes[0][idx][0]) is int):
                self.pld_indexes[0][idx] = self.pld_indexes[0][idx][ref_seg_a]
                self.pld_bins[0][idx] = self.pld_bins[0][idx][ref_seg_a]

            self.rec_indexes[0].append(self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])])
            self.rec_bins[0].append(np.max(self.pld_bins[0][idx]))

        # preamble of A is overlapped by preamble or SFD of B
        for idx in range(max(self.detect_chirp_offset - 12, 0), min(self.pld_chirps, self.detect_chirp_offset)):
            curr_index = []
            curr_max = []

            if not (type(self.pld_indexes[0][idx][0]) is int):
                self.pld_indexes[0][idx] = self.pld_indexes[0][idx][ref_seg_a]
                self.pld_bins[0][idx] = self.pld_bins[0][idx][ref_seg_a]

            # excluding preamble chirp index of B
            for jdx in range(len(self.pld_indexes[0][idx])):
                if abs(self.pld_indexes[0][idx][jdx] - preamble_index) >= self.pld_tolerance \
                        and self.pld_bins[0][idx][jdx] > self.bin_threshold:
                    curr_index.append(self.pld_indexes[0][idx][jdx])
                    curr_max.append(self.pld_bins[0][idx][jdx])

            if len(curr_index) >= 1:
                self.rec_indexes[0].append(curr_index[np.argmax(curr_max)])
                self.rec_bins[0].append(np.max(curr_max))
            else:
                self.rec_indexes[0].append(self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])])
                self.rec_bins[0].append(np.max(self.pld_bins[0][idx]))

        # chirp of A is partly overlapped by the first payload chirp of B
        if self.detect_chirp_offset < self.pld_chirps:
            if not (type(self.pld_indexes[0][self.detect_chirp_offset][0]) is int):
                self.rec_indexes[0].append(self.pld_indexes[0][self.detect_chirp_offset][ref_seg_a])
                self.rec_bins[0].append(self.pld_bins[0][self.detect_chirp_offset][ref_seg_a])
            else:
                self.rec_indexes[0].append(self.pld_indexes[0][self.detect_chirp_offset])
                self.rec_bins[0].append(self.pld_bins[0][self.detect_chirp_offset])

        # payload chirp of A is overlapped by two payload chirps of B
        for idx in range(min(self.pld_chirps, self.detect_chirp_offset + 1), self.pld_chirps):
            # using the largest segment of A as reference
            curr_idx = self.pld_indexes[0][idx][ref_seg_a]
            curr_max = self.pld_bins[0][idx][ref_seg_a]

            # counting the number of repeated index
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[0][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[0][idx][jdx][kdx]) <= self.pld_tolerance and \
                                self.pld_bins[0][idx][jdx][kdx] > 0:
                            curr_cnt[ldx] += 1
                            # curr_idx[ldx].append(pld_indexes[0][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]

            if len(rpt_index) == 1:
                # only one repeated index
                self.rec_indexes[0].append(curr_idx[rpt_index[0]])
                self.rec_bins[0].append(curr_max[rpt_index[0]])
            else:
                # save multiple repeated indexes
                self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        samp_offset_b = {1: self.d_samples_per_symbol - self.detect_samp_offset, 2: self.detect_samp_offset}
        samp_offset_b = sorted(samp_offset_b.items(), key=lambda x: x[1])
        ref_seg_b = samp_offset_b[-1][0]

        # payload chirp of B is overlapped by two payload chirps of A
        for idx in range(0, max(0, self.pld_chirps - self.detect_chirp_offset - 1)):
            # using longest segment of B as reference
            curr_idx = self.pld_indexes[1][idx][ref_seg_b]
            curr_max = self.pld_bins[1][idx][ref_seg_b]

            # counting the number of repeated index
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            # curr_idx[ldx].append(pld_indexes[1][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]

            if len(rpt_index) == 1:
                # only one repeated index
                self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                self.rec_bins[1].append(curr_max[rpt_index[0]])
            else:
                # save multiple repeated indexes
                self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # chirp of B is partly overlapped by the last payload chirp of A
        if self.detect_chirp_offset < self.pld_chirps:
            if not (type(self.pld_indexes[1][self.pld_chirps - self.detect_chirp_offset - 1][0]) is int):
                self.rec_indexes[1].append(self.pld_indexes[1][self.pld_chirps - self.detect_chirp_offset - 1][ref_seg_b])
                self.rec_bins[1].append(self.pld_bins[1][self.pld_chirps - self.detect_chirp_offset - 1][ref_seg_b])
            else:
                self.rec_indexes[1].append(self.pld_indexes[1][self.pld_chirps - self.detect_chirp_offset - 1])
                self.rec_bins[1].append(self.pld_bins[1][self.pld_chirps - self.detect_chirp_offset - 1])

        # payload of B is collision free
        for idx in range(max(self.pld_chirps - self.detect_chirp_offset, 0), self.pld_chirps):
            self.rec_indexes[1].append(self.pld_indexes[1][idx][np.argmax(self.pld_bins[1][idx])])
            self.rec_bins[1].append(np.max(self.pld_bins[1][idx]))

        # symbol recovery based on repeated index is failed for A
        for idx in range(self.detect_chirp_offset, self.pld_chirps):
            if not (type(self.rec_indexes[0][idx]) is int):
                curr_idx = []
                curr_bin = []
                oly_idx = []

                oly_pld_b = idx - self.detect_chirp_offset - 2 + ref_seg_a
                # payload chirp index of B when aligning with chirp of A
                if 0 <= oly_pld_b < self.pld_chirps:
                    if type(self.rec_indexes[1][oly_pld_b]) is int:
                        oly_idx.append(
                            (self.rec_indexes[1][oly_pld_b] - self.detect_bin_offset) % self.d_number_of_bins)
                else:
                    self.rec_indexes[0][idx] = self.rec_indexes[0][idx][np.argmax(self.rec_bins[0][idx])]
                    self.rec_bins[0][idx] = np.max(self.rec_bins[0][idx])

                    continue

                # excluding payload chirp index of B
                for jdx in range(len(self.rec_indexes[0][idx])):
                    is_oly = False
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[0][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[0][idx][jdx])
                        curr_bin.append(self.rec_bins[0][idx][jdx])

                if len(curr_idx) == 1:
                    # cross decoding successes
                    self.rec_indexes[0][idx] = curr_idx[0]
                    self.rec_bins[0][idx] = curr_bin
                else:
                    # cross decoding is also failed
                    # performing power mapping
                    if self.sfd_bins[0] > self.sfd_bins[1]:
                        jdx = 0
                    else:
                        jdx = 1

                    self.rec_indexes[0][idx] = self.rec_indexes[0][idx][jdx]
                    self.rec_bins[0][idx] = self.rec_bins[0][idx][jdx]

        # symbol recovery based on repeated index is failed for B
        for idx in range(0, self.pld_chirps - self.detect_chirp_offset):
            if not (type(self.rec_indexes[1][idx]) is int):
                curr_idx = []
                curr_bin = []
                oly_idx = []

                oly_pld_a = idx + self.detect_chirp_offset - 1 + ref_seg_b
                # payload chirp index of B when aligning with chirp of A
                if self.detect_chirp_offset <= oly_pld_a < self.pld_chirps:
                    if type(self.rec_indexes[0][oly_pld_a]) is int:
                        oly_idx.append(
                            (self.rec_indexes[0][oly_pld_a] + self.detect_bin_offset) % self.d_number_of_bins)
                else:
                    self.rec_indexes[1][idx] = self.rec_indexes[1][idx][np.argmax(self.rec_bins[1][idx])]
                    self.rec_bins[1][idx] = np.max(self.rec_bins[1][idx])

                    continue

                # excluding payload chirp index of A
                for jdx in range(len(self.rec_indexes[1][idx])):
                    is_oly = False
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[1][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[1][idx][jdx])
                        curr_bin.append(self.rec_bins[1][idx][jdx])

                if len(curr_idx) == 1:
                    # cross decoding successes
                    self.rec_indexes[1][idx] = curr_idx[0]
                    self.rec_bins[1][idx] = curr_bin[0]
                else:
                    # cross decoding is also failed
                    # performing power mapping
                    if self.sfd_bins[0] > self.sfd_bins[1]:
                        jdx = 1
                    else:
                        jdx = 0

                    self.rec_indexes[1][idx] = self.rec_indexes[1][idx][jdx]
                    self.rec_bins[1][idx] = self.rec_bins[1][idx][jdx]

        # recording the payload chirp index of B when aligning with A
        self.oly_indexes.append(np.array(
            np.mod(np.add(self.rec_indexes[1], self.d_number_of_bins - self.detect_bin_offset), self.d_number_of_bins),
            dtype=np.int))
        # recording the payload chirp index of A when aligning with B
        self.oly_indexes.append(np.array(
            np.mod(np.add(self.rec_indexes[0], self.d_number_of_bins + self.detect_bin_offset), self.d_number_of_bins),
            dtype=np.int))

    def error_correct(self):
        for idx in range(len(self.rec_indexes)):
            if len(self.rec_indexes[idx]) == self.pld_chirps:
                corr_idx = []
                idx_error = 0

                for jdx in range(8):
                    ogn_idx = self.rec_indexes[idx][jdx]
                    curr_idx = np.int16(np.rint(ogn_idx / 4.0))

                    idx_error += self.packet_header[self.packet_index][jdx] - ogn_idx
                    corr_idx.append(4 * curr_idx)

                idx_error = np.rint(idx_error / 8.0)

                if abs(idx_error) >= 2:
                    idx_error = 0

                for jdx in range(8, self.pld_chirps):
                    ogn_idx = self.rec_indexes[idx][jdx]
                    curr_idx = np.int16((ogn_idx + idx_error) % self.d_number_of_bins)
                    corr_idx.append(curr_idx)

                self.corr_indexes.append(corr_idx)

    def get_chirp_error(self):

        for idx in range(len(self.corr_indexes)):
            errors_list = np.abs(
                np.subtract(self.packet_chirp[self.packet_index], np.array([self.corr_indexes[idx]] * 2)))
            errors = [np.count_nonzero(errors_list[0]), np.count_nonzero(errors_list[1])]

            self.chirp_errors.append(np.min(errors))

    def recover_byte(self):
        for idx in range(len(self.corr_indexes)):
            self.rec_bytes.append(self.lora_decoder.decode(self.corr_indexes[idx].copy()))

    def get_byte_error(self):
        for idx in range(len(self.rec_bytes)):
            comp_result_0 = []
            comp_result_1 = []
            header_result = []

            for jdx in range(3):
                header_result.append(
                    bin(self.packet_byte[self.packet_index][idx][jdx] ^ self.rec_bytes[idx][jdx]).count('1'))

            for jdx in range(0, len(self.rec_bytes[idx])):
                comp_result_0.append(
                    bin(self.packet_byte[self.packet_index][0][jdx] ^ self.rec_bytes[idx][jdx]).count('1'))
                comp_result_1.append(
                    bin(self.packet_byte[self.packet_index][1][jdx] ^ self.rec_bytes[idx][jdx]).count('1'))

            self.header_errors.append(np.sum(header_result))
            self.byte_errors.append(np.min([np.sum(comp_result_0), np.sum(comp_result_1)]))

    def show_result(self):
        print('Preamble Success')
        print('Preamble Indexes: ', self.preamble_indexes)
        print('Preamble Bins   : ', self.preamble_bins)
        print('Preamble Begins : ', self.pre_begins)

        print('SFD      Success')
        print('SFD      Indexes: ', self.sfd_indexes)
        print('SFD      Bins   : ', self.sfd_bins)
        print('SFD      Begins : ', self.sfd_cnts)

        print('Decoding Result:')
        print('Correct   Index:', self.corr_indexes)
        print('Recover   Bytes:', self.rec_bytes)
        print('Chirp    Errors:', self.chirp_errors)
        print('Header   Errors:', self.header_errors)
        print('Byte     Errors:', self.byte_errors)

    def save_result(self):

        self.pre_idxes_list.append(self.preamble_indexes)
        self.pre_bins_list.append(self.preamble_indexes)
        self.pre_bgns_list.append(self.pre_begins)

        self.sfd_idxes_list.append(self.sfd_indexes)
        self.sfd_bins_list.append(self.sfd_bins)
        self.sfd_cnts_list.append(self.sfd_cnts)

        self.detect_offsets_list.append([self.detect_chirp_offset, self.detect_samp_offset, self.detect_offset])

        self.corr_idxes_list.append(self.corr_indexes.copy())
        self.rec_bytes_list.append(self.rec_bytes.copy())
        self.chirp_errors_list.append(self.chirp_errors)
        self.header_errors_list.append(self.header_errors)
        self.byte_errors_list.append(self.byte_errors)


if __name__ == "__main__":
    test_bw = 125e3
    test_samples_per_second = 1e6
    test_sf = 7
    test_cr = 4
    test_powers = [2, 0]
    test_byte_lens = [8, 8]
    test_chirp_lens = [44, 44]

    offresult2 = OffResult()

    mlora2 = MLoRa2Online(test_bw, test_samples_per_second, test_sf, test_cr, test_powers, test_byte_lens, test_chirp_lens)
    mlora2.read_packets()
    mlora2.build_ideal_chirp()

    # lora2 = LoRa2(test_bw, test_samples_per_second, test_sf, test_cr, test_powers, test_lengths)
    # lora2.read_packets()
    # lora2.build_ideal_chirp()

    bgn_postition = np.random.randint(0, mlora2.d_samples_per_symbol)
    i = bgn_postition
    mlora2.init_decoding()
    while i < len(mlora2.packets_mlora2):
        curr_chirp = mlora2.packets_mlora2[i:i+mlora2.d_samples_per_symbol]
        mlora2.decoding_packet(curr_chirp)

        i += mlora2.d_samples_per_symbol


