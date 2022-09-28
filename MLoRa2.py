import pickle as pl
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from cmath import phase
from scipy.fftpack import fft
from Stack import Stack, State
from lora_decode import LoRaDecode
from LoRa2 import LoRa2
from OffResult import OffResult

debug_off = -1
debug_info = 1
debug_verbose = 2
debug_on = debug_off

class MLoRa2:

    def __init__(self, bw, samples_per_second, sf, cr, powers, lengths):
        # Debugging
        self.d_debug = debug_on

        # Read all packets
        self.packets = None
        self.packets_shr = None
        self.d_powers = powers
        self.d_lengths = lengths
        self.packet_index = np.int(np.log2(lengths[0])) - 3
        self.payload_length = lengths[0]
        self.packet_length = np.int(lengths[0] + 3)

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

        self.packet_chirp =\
                     [[[124,  0,   0,  76,  4, 108,   8, 120,
                         1, 18, 67, 90, 66, 15, 31, 38,
                         111, 2, 10, 108, 27, 42, 27, 33,
                         50, 58, 80, 91, 32, 79, 57, 85],
                        [124,  0,   0,  76,  4, 108,   8, 120,
                         84, 71, 94, 85, 5, 122, 30, 123,
                         58, 87, 49, 115, 106, 63, 24, 100,
                         48, 68, 16, 123, 48, 71, 61, 87]],
                      [[56,   0, 28, 124, 116,  96, 108, 40,
                        1, 18, 67, 90, 66, 15, 31, 38,
                        111, 2, 10, 108, 27, 42, 27, 33,
                        106, 17, 64, 80, 91, 109, 103, 18,
                        1, 117, 83, 5, 53, 35, 119, 99,
                        54, 66, 47, 69, 111, 64, 49, 92],
                       [56,   0, 28, 124, 116,  96, 108, 40,
                        84, 71, 94, 85, 5, 122, 30, 123,
                        58, 87, 49, 115, 106, 63, 24, 100,
                        63, 68, 8, 112, 73, 70, 99, 103,
                        84, 32, 61, 69, 17, 116, 127, 118,
                        60, 56, 45, 59, 47, 104, 33, 86]]]

        self.packet_byte = np.array(
                           [[[8, 128, 144, 18, 52, 86, 120, 18, 52, 86, 120, 0, 0, 0, 0, 0, 0, 0, 0],
                             [8, 128, 144, 33, 67, 101, 135,33, 67, 101, 135, 0, 0, 0, 0, 0, 0, 0, 0]],
                            [[16, 129, 64, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120],
                             [16, 129, 64, 33, 67, 101, 135,33, 67, 101, 135, 33, 67, 101, 135,33, 67, 101, 135]]],
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

        self.packet = None
        self.packet_shr = None
        self.packet_o = None

        # Initialize Decoding
        self.preamble_peak = 3
        self.bin_threshold = 9
        self.preamble_chirps = 6
        self.preamble_tolerance = 2
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_history = []
        self.prebin_history = []
        for idx in range(3):
            self.pre_history.append(Stack())
            self.prebin_history.append(Stack())

        self.pre_ordered = False
        self.pre_sfd_dist = 4
        self.sfd_dist = [0, 0]
        self.sfd_peak = 3
        self.sfd_chirps = 2
        self.sfd_tolerance = 2
        self.sfd_indexes = []
        self.sfd_bins = []
        self.sfd_begins = []
        self.sfd_history = []
        self.sfdbin_history = []
        self.sfdbins_history = []

        self.pld_peak = 2
        self.detect_offset = None
        self.detect_chirp_offset = None
        self.detect_samp_offset = None
        self.detect_bin_offset = None
        self.shift_samples = []
        self.pld_chirps = int(8 + (4 + self.d_cr) * np.ceil(self.d_lengths[0] * 2.0 / self.d_sf))
        self.pld_tolerance = 2
        self.pld_indexes = [[], []]
        self.pld_bins = [[], []]

        self.lora_decoder = LoRaDecode(self.payload_length, self.d_sf, self.d_cr)
        self.rec_indexes = [[], []]
        self.rec_bins = [[], []]
        self.oly_indexes = []
        self.corr_indexes = [[], []]
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_states = [State.S_RESET, State.S_RESET]

        # Save result
        self.detected_offsets = [[], [], []]
        self.decoded_prebins = [[], []]
        self.decoded_sfdbins = [[], []]
        self.decoded_chirps = [[], []]
        self.decoded_bytes = [[], []]
        self.decoded_chirp_errors = [[], []]
        self.decoded_header_errors = [[], []]
        self.decoded_byte_errors = [[], []]

    def read_packets(self):
        path_prefix = "../dbgresult/FFT05051921/"
        exp_prefix = ["DIST/", "PWR/", "SF/", "PWR_LEN/"]

        power = self.d_powers
        length = self.d_lengths
        sf = [self.d_sf, self.d_sf]
        payload = ['fwd', 'inv']

        exp_idx = 3

        file_name = [
            path_prefix + exp_prefix[exp_idx] + "data_" + payload[0] +
            "_sf_" + str(sf[0]) + "_len_" + str(length[0]) + "_pwr_" + str(power[0]) + ".mat",
            path_prefix + exp_prefix[exp_idx] + "data_" + payload[1] +
            "_sf_" + str(sf[1]) + "_len_" + str(length[1]) + "_pwr_" + str(power[1]) + ".mat"]

        data_0 = sio.loadmat(file_name[0])
        data_1 = sio.loadmat(file_name[1])

        self.packets = [data_0['packets'], data_1['packets']]
        self.packets_shr = [data_0['packets_shr'], data_1['packets_shr']]

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

    def read_packet_shr(self, packet_idx):
        # all time offset space
        # self.chirp_offset = np.random.randint(0, self.chirp_cnt)
        # self.samp_offset = np.random.randint(0, self.d_samples_per_symbol)
        # self.time_offset = self.chirp_offset * self.d_samples_per_symbol + self.samp_offset

        # controlled time offset space
        self.chirp_cnt = len(self.packets_shr[0][0]) // self.d_samples_per_symbol
        self.detect_scope = 2
        self.chirp_offset = np.random.randint(0, self.chirp_cnt - 1)
        self.samp_offset = self.d_decim_factor * np.random.randint(self.detect_scope + 1, self.d_number_of_bins - self.detect_scope)

        # configured time offset
        # self.chirp_offset = 0
        # self.samp_offset = 125 * self.d_decim_factor

        self.time_offset = self.chirp_offset * self.d_samples_per_symbol + self.samp_offset

        self.packet_idx = packet_idx
        self.packet = [self.packets[0][packet_idx], self.packets[1][packet_idx]]
        self.packet_shr = [self.packets_shr[0][packet_idx], self.packets_shr[1][packet_idx]]

        self.packet_o = self.packet_shr[0][:self.time_offset]
        self.packet_o = np.concatenate((self.packet_o, np.add(self.packet_shr[0][self.time_offset:], self.packet_shr[1][:len(self.packet_shr[1]) - self.time_offset])))
        self.packet_o = np.concatenate((self.packet_o, self.packet_shr[1][len(self.packet_shr[1]) - self.time_offset:]))

        if self.d_debug >= 1:
            self.show_info()


    def init_decoding(self):
        self.bin_threshold = 9
        self.preamble_chirps = 6
        self.preamble_tolerance = 2
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_history = []
        self.prebin_history = []
        for idx in range(3):
            self.pre_history.append(Stack())
            self.prebin_history.append(Stack())

        self.pre_ordered = False
        self.pre_sfd_dist = 4
        self.sfd_dist = [0, 0]
        self.sfd_chirps = 2
        self.sfd_tolerance = 2
        self.sfd_indexes = []
        self.sfd_bins = []
        self.sfd_begins = []
        self.sfd_history = []
        self.sfdbin_history = []
        self.sfdbins_history = []

        # self.detect_offset = self.time_offset
        self.shift_samples = []
        self.pld_chirps = int(8 + (4 + self.d_cr) * np.ceil(self.d_lengths[0] * 2.0 / self.d_sf))
        self.pld_tolerance = 2
        self.pld_indexes = [[], []]
        self.pld_bins = [[], []]

        self.rec_indexes = [[], []]
        self.rec_bins = [[], []]
        self.oly_indexes = []
        self.corr_indexes = [[], []]
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

    def decoding_packet(self):
        idx = 0
        while idx<len(self.packet_o):
            if idx + self.d_samples_per_symbol > len(self.packet_o):
                break

            bgn_index = idx
            end_index = idx + self.d_samples_per_symbol
            chirp_o = self.packet_o[bgn_index:end_index]
            chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp_o, self.d_samples_per_symbol, self.detect_scope, self.preamble_peak)

            for jdx in range(3):
                self.pre_history[jdx].push(chirp_index[jdx])
                self.prebin_history[jdx].push(chirp_max[jdx])

                if self.pre_history[jdx].size() > self.preamble_chirps:
                    self.pre_history[jdx].pop_back()
                    self.prebin_history[jdx].pop_back()

            if self.d_states[0] == State.S_PREFILL and self.d_states[1] == State.S_PREFILL:
                if self.pre_history[0].size() >= self.preamble_chirps:
                    for jdx in range(2):
                        self.d_states[jdx] = State.S_DETECT_PREAMBLE
                else:
                    idx += self.d_samples_per_symbol
                    continue

            if self.d_states[0] == State.S_DETECT_PREAMBLE or self.d_states[1] == State.S_DETECT_PREAMBLE:
                detect_indexes, detect_bins = self.detect_preamble_all(self.pre_history, self.prebin_history)
                # detect_indexes, detect_bins = self.detect_preamble(self.pre_history, self.prebin_history)
                if self.d_states[0] == State.S_DETECT_PREAMBLE and self.d_states[1] == State.S_DETECT_PREAMBLE:
                    if len(np.where(detect_indexes >= 0)[0]) > 0:
                        # sort detected preamble indexes according to the arrival order
                        detect_dicts = dict(zip(detect_indexes, detect_bins))
                        detect_dicts = sorted(detect_dicts.items(), key=lambda x: x[0])
                        detect_indexes.sort()
                        repeat_index = np.where(detect_indexes >= 0)[0]
                        if len(repeat_index) >= 2:
                            for jdx in range(2):
                                self.d_states[jdx] = State.S_SFD_SYNC
                                self.preamble_indexes.append(detect_dicts[repeat_index[jdx]][0])
                                self.preamble_bins.append(detect_dicts[repeat_index[jdx]][1])

                            # align with the first packet
                            # i -= preamble_indexes[0] * d_decim_factor
                            # preamble_indexes[1] += preamble_indexes[0]
                        else:
                            self.d_states[0] = State.S_SFD_SYNC
                            self.preamble_indexes.append(detect_dicts[repeat_index[0]][0])
                            self.preamble_bins.append(detect_dicts[repeat_index[0]][1])
                            # align with the first packet
                            # i -= preamble_indexes[0] * d_decim_factor

                if self.d_states[0] != State.S_DETECT_PREAMBLE and self.d_states[1] == State.S_DETECT_PREAMBLE:
                    unique_index = np.where((detect_indexes >= 0) & (detect_indexes != self.preamble_indexes[0]))[0]
                    repeat_index = np.where(detect_indexes >= 0)[0]
                    if len(unique_index) > 0 and self.d_states[0] == State.S_SFD_SYNC and abs(detect_indexes[unique_index[0]] - self.preamble_indexes[0]) >= self.preamble_tolerance:
                        self.d_states[1] = State.S_SFD_SYNC
                        self.preamble_indexes.append(detect_indexes[unique_index[0]])
                        self.preamble_bins.append(detect_bins[unique_index[0]])

                    if len(repeat_index) > 0 and self.d_states[0] != State.S_SFD_SYNC:
                        self.d_states[1] = State.S_SFD_SYNC
                        self.preamble_indexes.append(detect_indexes[repeat_index[0]])
                        self.preamble_bins.append(detect_bins[repeat_index[0]])

            if self.d_states[0] == State.S_SFD_SYNC or self.d_states[1] == State.S_SFD_SYNC:
                if not self.pre_ordered:
                    pre_dicts = dict(zip(self.preamble_indexes, self.preamble_bins))
                    pre_dicts = sorted(pre_dicts.items(), key=lambda x: x[0])

                    self.preamble_indexes = [pre_dict[0] for pre_dict in pre_dicts]
                    self.preamble_bins = [pre_dict[1] for pre_dict in pre_dicts]

                if self.d_states[0] == State.S_SFD_SYNC and self.d_states[1] != State.S_SFD_SYNC:
                    self.sfd_dist[0] += 1
                    bgn_sfd = idx - self.preamble_indexes[0] * self.d_decim_factor
                    if not self.detect_preamble_chirp(0, idx, self.detect_scope, self.sfd_peak) \
                            and self.detect_down_chirp(bgn_sfd, self.detect_scope, self.sfd_peak):
                        self.d_states[0] = State.S_READ_PAYLOAD
                        # i += 2*d_samples_per_symbol + d_quad_samples_per_symbol - preamble_indexes[0] * d_decim_factor
                        # Record shift samples needed to align with the first packet
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                             - self.preamble_indexes[0] * self.d_decim_factor)
                        self.sfd_begins.append(bgn_sfd)

                if self.d_states[0] != State.S_SFD_SYNC and self.d_states[1] == State.S_SFD_SYNC:
                    self.sfd_dist[1] += 1
                    bgn_sfd = idx - self.preamble_indexes[1] * self.d_decim_factor
                    # add preamble to sfd distance to avoid detecting sfd of the first packet
                    if not self.detect_preamble_chirp(1, idx, self.detect_scope, self.sfd_peak) \
                            and self.detect_down_chirp(bgn_sfd, self.detect_scope, self.sfd_peak):
                        self.d_states[1] = State.S_READ_PAYLOAD
                        # Record shift samples need to align with the second packet
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                             - self.preamble_indexes[1] * self.d_decim_factor)
                        self.sfd_begins.append(bgn_sfd)

                if self.d_states[0] == State.S_SFD_SYNC and self.d_states[1] == State.S_SFD_SYNC:
                    self.sfd_dist[0] += 1
                    bgn_sfd = idx - self.preamble_indexes[0] * self.d_decim_factor
                    if not self.detect_preamble_chirp(0, idx, self.detect_scope, self.sfd_peak) \
                            and self.detect_down_chirp(bgn_sfd, self.detect_scope, self.sfd_peak):
                        self.d_states[0] = State.S_READ_PAYLOAD
                        # i += 2*d_samples_per_symbol + d_quad_samples_per_symbol - preamble_indexes[0] * d_decim_factor
                        # Record shift samples needed to align with the first packet
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                             - self.preamble_indexes[0] * self.d_decim_factor)
                        self.sfd_begins.append(bgn_sfd)

                    self.sfd_dist[1] += 1
                    bgn_sfd = idx - self.preamble_indexes[1] * self.d_decim_factor
                    # add preamble to sfd distance to avoid detecting sfd of the first packet
                    if not self.detect_preamble_chirp(1, idx, self.detect_scope, self.sfd_peak) \
                            and self.detect_down_chirp(bgn_sfd, self.detect_scope, self.sfd_peak):
                        self.d_states[1] = State.S_READ_PAYLOAD
                        # Record shift samples need to align with the second packet
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                             - self.preamble_indexes[1] * self.d_decim_factor)
                        self.sfd_begins.append(bgn_sfd)


            if self.d_states[0] == State.S_READ_PAYLOAD or self.d_states[1] == State.S_READ_PAYLOAD:

                if self.d_states[0] == State.S_READ_PAYLOAD and self.d_states[1] != State.S_READ_PAYLOAD:

                    if len(self.pld_indexes[0]) < self.pld_chirps:
                        chirp_0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

                        bgn_index_0 = idx + self.shift_samples[0]
                        end_index_0 = bgn_index_0 + self.d_samples_per_symbol
                        chirp_0[0:len(self.packet_o[bgn_index_0:end_index_0])] = self.packet_o[bgn_index_0:end_index_0]

                        chirp_index_0, chirp_max_0, chirp_bin_0 = self.get_fft_bins(chirp_0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                        # chirp_index_0 = [chirp_index_0[j] for j in np.where(np.array(chirp_max_0) > bin_threshold)[0]]
                        # chirp_max_0 = [chirp_max_0[j] for j in np.where(np.array(chirp_max_0) > bin_threshold)[0]]

                        self.pld_indexes[0].append(chirp_index_0)
                        self.pld_bins[0].append(chirp_max_0)

                        # if d_debug == 1:
                        #     show_signal([chirp_bin_0, np.zeros(d_number_of_bins)],
                        #                 'Oly FFT 0 with index ' + str(len(pld_indexes[0]) - 1))
                        #     show_chirp(bgn_index_0, end_index_0, 0)

                    if len(self.pld_indexes[0]) >= self.pld_chirps:
                        self.d_states[0] = State.S_STOP

                if self.d_states[0] != State.S_READ_PAYLOAD and self.d_states[1] == State.S_READ_PAYLOAD:

                    if len(self.pld_indexes[1]) < self.pld_chirps:
                        # pre-assign to avoid the rest samples cannot form a complete chirp
                        chirp_1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

                        bgn_index_1 = idx + self.shift_samples[1]
                        end_index_1 = bgn_index_1 + self.d_samples_per_symbol

                        chirp_1[0:len(self.packet_o[bgn_index_1:end_index_1])] = self.packet_o[bgn_index_1:end_index_1]

                        chirp_index_1, chirp_max_1, chirp_bin_1 = self.get_fft_bins(chirp_1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                        # chirp_index_1 = [chirp_index_1[j] for j in np.where(np.array(chirp_max_1) > threshold)[0]]
                        # chirp_max_1 = [chirp_max_1[j] for j in np.where(np.array(chirp_max_1) > threshold)[0]]

                        self.pld_indexes[1].append(chirp_index_1)
                        self.pld_bins[1].append(chirp_max_1)

                    if len(self.pld_indexes[1]) >= self.pld_chirps:
                        self.d_states[1] = State.S_STOP

                if self.d_states[0] == State.S_READ_PAYLOAD and self.d_states[1] == State.S_READ_PAYLOAD:
                    self.detect_offset = self.sfd_begins[1] - self.sfd_begins[0]
                    self.detect_chirp_offset = self.detect_offset // self.d_samples_per_symbol
                    self.detect_samp_offset = self.detect_offset % self.d_samples_per_symbol
                    self.detect_bin_offset = self.detect_samp_offset // self.d_decim_factor

                    if len(self.pld_indexes[0]) < self.pld_chirps:
                        bgn_index_0 = idx + self.shift_samples[0]
                        end_index_0 = bgn_index_0 + self.d_samples_per_symbol
                        chirp_0 = self.packet_o[bgn_index_0:end_index_0]
                        chirp_p0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                        chirp_b0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

                        chirp_p0[0:self.detect_samp_offset] = chirp_0[0:self.detect_samp_offset]
                        chirp_b0[self.detect_samp_offset:self.d_samples_per_symbol] = chirp_0[self.detect_samp_offset:self.d_samples_per_symbol]


                        # if self.detect_samp_offset >= (self.d_samples_per_symbol - self.detect_samp_offset):
                        #     chirp_p0[0:self.detect_samp_offset] = chirp_0[0:self.detect_samp_offset]
                        #     chirp_b0[self.d_half_samples_per_symbol:self.d_samples_per_symbol] \
                        #         = chirp_0[self.d_half_samples_per_symbol:self.d_samples_per_symbol]
                        # else:
                        #     chirp_p0[0:self.d_half_samples_per_symbol] = chirp_0[0:self.d_half_samples_per_symbol]
                        #     chirp_b0[self.detect_samp_offset:self.d_samples_per_symbol] = chirp_0[self.detect_samp_offset:self.d_samples_per_symbol]

                        chirp_index_0, chirp_max_0, chirp_bin_0 = self.get_fft_bins(chirp_0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
                        chirp_index_p0, chirp_max_p0, chirp_bin_p0 = self.get_fft_bins(chirp_p0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
                        chirp_index_b0, chirp_max_b0, chirp_bin_b0 = self.get_fft_bins(chirp_b0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                        # chirp_index_0 = [chirp_index_0[j] for j in np.where(np.array(chirp_max_0) > threshold)[0]]
                        # chirp_max_0 = [chirp_max_0[j] for j in np.where(np.array(chirp_max_0) > threshold)[0]]

                        self.pld_indexes[0].append([chirp_index_0, chirp_index_p0, chirp_index_b0])
                        self.pld_bins[0].append([chirp_max_0, chirp_max_p0, chirp_max_b0])

                        # if self.d_debug == 1:
                        #     if len(pld_indexes[0]) == pld_chirps + 1:
                        #         show_signal([chirp_bin_0, np.zeros(d_number_of_bins)],
                        #                     'Oly FFT 0 with index ' + str(len(pld_indexes[0]) - 1))
                        #         show_signal([chirp_bin_p0, chirp_bin_b0],
                        #                     'Oly FFT 0 with index ' + str(len(pld_indexes[0]) - 1))
                        #         show_chirp(bgn_index_0, end_index_0, 0)

                    if len(self.pld_indexes[0]) >= self.pld_chirps:
                        self.d_states[0] = State.S_STOP

                    if len(self.pld_indexes[1]) < self.pld_chirps:
                        bgn_index_1 = idx + self.shift_samples[1]
                        end_index_1 = bgn_index_1 + self.d_samples_per_symbol
                        chirp_1 = self.packet_o[bgn_index_1:end_index_1]
                        chirp_p1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
                        chirp_b1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

                        chirp_p1[0:self.d_samples_per_symbol - self.detect_samp_offset] \
                            = chirp_1[0:self.d_samples_per_symbol - self.detect_samp_offset]
                        chirp_b1[self.d_samples_per_symbol - self.detect_samp_offset:self.d_samples_per_symbol] = \
                            chirp_1[self.d_samples_per_symbol - self.detect_samp_offset:self.d_samples_per_symbol]

                        # if self.detect_samp_offset >= (self.d_samples_per_symbol - self.detect_samp_offset):
                        #     chirp_p1[0:self.d_half_samples_per_symbol] = chirp_1[0:self.d_half_samples_per_symbol]
                        #     chirp_b1[self.d_samples_per_symbol - self.detect_samp_offset:self.d_samples_per_symbol] = \
                        #         chirp_1[self.d_samples_per_symbol - self.detect_samp_offset:self.d_samples_per_symbol]
                        # else:
                        #     chirp_p1[0:self.d_samples_per_symbol - self.detect_samp_offset] \
                        #         = chirp_1[0:self.d_samples_per_symbol - self.detect_samp_offset]
                        #     chirp_b1[self.d_half_samples_per_symbol:self.d_samples_per_symbol] = \
                        #         chirp_1[self.d_half_samples_per_symbol:self.d_samples_per_symbol]

                        chirp_index_1, chirp_max_1, chirp_bin_1 = self.get_fft_bins(chirp_1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
                        chirp_index_p1, chirp_max_p1, chirp_bin_p1 = self.get_fft_bins(chirp_p1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
                        chirp_index_b1, chirp_max_b1, chirp_bin_b1 = self.get_fft_bins(chirp_b1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

                        # chirp_index_1 = [chirp_index_1[j] for j in np.where(np.array(chirp_max_1) > threshold)[0]]
                        # chirp_max_1 = [chirp_max_1[j] for j in np.where(np.array(chirp_max_1) > threshold)[0]]

                        self.pld_indexes[1].append([chirp_index_1, chirp_index_p1, chirp_index_b1])
                        self.pld_bins[1].append([chirp_max_1, chirp_max_p1, chirp_max_b1])

                        # if self.d_debug == 1:
                        #     if len(pld_indexes[1]) == pld_chirps + 1:
                        #         show_signal([chirp_bin_1, np.zeros(d_number_of_bins)],
                        #                     'Oly FFT 1 with index ' + str(len(pld_indexes[1]) - 1))
                        #         show_signal([chirp_bin_p1, chirp_bin_b1],
                        #                     'Oly FFT 1 with index ' + str(len(pld_indexes[1]) - 1))
                        #         show_chirp(bgn_index_1, end_index_1, 1)

                    if len(self.pld_indexes[1]) >= self.pld_chirps:
                        self.d_states[1] = State.S_STOP

            if self.d_states[0] == State.S_STOP and self.d_states[1] == State.S_STOP:
                self.detect_offset = self.sfd_begins[1] - self.sfd_begins[0]
                self.detect_chirp_offset = self.detect_offset // self.d_samples_per_symbol
                self.detect_samp_offset = self.detect_offset % self.d_samples_per_symbol
                self.detect_bin_offset = self.detect_samp_offset // self.d_decim_factor

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
        # self.show_result()

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
        curr_decim_fft = np.add(curr_fft[:self.d_number_of_bins], curr_fft[self.d_samples_per_symbol - self.d_number_of_bins:])

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
            curr_peaks.update({-(len(curr_peaks)+1):-(len(curr_peaks)+1)})
        sorted_peaks = sorted(curr_peaks.items(), key=lambda kv: kv[1])
        max_idx = [sorted_peaks[-idx][0] for idx in range(1, num + 1)]
        max_bin = [sorted_peaks[-idx][1] for idx in range(1, num + 1)]
        # max_peaks = [(sorted_peaks[-idx][0], sorted_peaks[-idx][1]) for idx in range(3)]

        return max_idx, max_bin

    def detect_preamble_all(self, idx_stacks, bin_stacks):
        pre_idxes = np.array([-idx-1 for idx in range(len(idx_stacks))])
        pre_bins = np.full(len(idx_stacks), -1, dtype=float)
        for idx in range(len(idx_stacks)):
            pre_idx = idx_stacks[idx].bottom()
            pre_found = True

            curr_idx = []
            curr_pos = []
            for jdx in range(self.preamble_chirps):
                curr_found = False
                for kdx in range(len(idx_stacks)):
                    if abs(pre_idx - idx_stacks[kdx].get_i(jdx)) < self.preamble_tolerance and \
                            bin_stacks[kdx].get_i(jdx) > self.bin_threshold:
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
                pre_bins[idx] = np.average([bin_stacks[bin_idx[1]].get_i(bin_idx[0]) for bin_idx in curr_idx[1:]])

        return pre_idxes, pre_bins

    def detect_preamble(self, idx_stacks, bin_stacks):
        pre_idxes = np.full(len(idx_stacks), -1)
        pre_bins = np.full(len(idx_stacks), -1, dtype=float)
        for idx in range(len(idx_stacks)):
            pre_idx = idx_stacks[idx].bottom()
            pre_found = True

            for jdx in range(self.preamble_chirps):
                if abs(pre_idx - idx_stacks[idx].get_i(jdx)) >= self.preamble_tolerance \
                        or bin_stacks[idx].get_i(jdx) <= self.bin_threshold:
                    pre_found = False
                    break

            if pre_found:
                pre_idxes[idx] = pre_idx
                pre_bins[idx] = np.average(bin_stacks[idx].get_list())

        return pre_idxes, pre_bins

    def detect_preamble_chirp(self, pkt_idx, bgn_idx, scope, num):
        chirp_o = self.packet_o[bgn_idx:bgn_idx+self.d_samples_per_symbol]
        chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp_o, self.d_samples_per_symbol, scope, num)

        is_preamble_chirp = False

        for idx in range(len(chirp_index)):
            if chirp_max[idx] > 0 and abs(chirp_index[idx]-self.preamble_indexes[pkt_idx]) < self.preamble_tolerance:
                is_preamble_chirp = True
                break

        return is_preamble_chirp


    def detect_down_chirp(self, bgn_idx, scope, num):
        sfd_bgns = [self.d_samples_per_symbol * 10, self.d_samples_per_symbol * 10 + self.time_offset]

        self.sfd_history.clear()
        self.sfdbin_history.clear()
        self.sfdbins_history.clear()

        for idx in range(self.sfd_chirps):
            pad_chirp = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            curr_chirp = self.packet_o[bgn_idx + idx * self.d_samples_per_symbol:bgn_idx + (idx + 1) * self.d_samples_per_symbol]
            pad_chirp[0:len(curr_chirp)] = curr_chirp
            sfd_idx, sfd_max, sfd_bin = self.get_down_chirp_bin(pad_chirp, scope, num)
            self.sfd_history.append(sfd_idx)
            self.sfdbin_history.append(sfd_max)
            self.sfdbins_history.append(sfd_bin)

        sfd_found = True

        curr_idx = []
        for idx in range(self.sfd_chirps):
            curr_found = False
            for jdx in range(3):
                if abs(self.sfd_history[idx][jdx] - self.d_half_number_of_bins) <= self.sfd_tolerance \
                        and self.sfdbin_history[idx][jdx] > self.bin_threshold:
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
            self.sfd_indexes.append([self.sfd_history[bin_idx[0]][bin_idx[1]] for bin_idx in curr_idx])
            self.sfd_bins.append(np.sum([self.sfdbin_history[bin_idx[0]][bin_idx[1]] for bin_idx in curr_idx]))

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
        # preamble chirp index of B when aligning with chirp of A
        preamble_shift = (self.d_quad_samples_per_symbol - self.preamble_indexes[0] * self.d_decim_factor) // self.d_decim_factor
        preamble_index = (self.preamble_indexes[1] + preamble_shift + self.d_number_of_bins) % self.d_number_of_bins

        # preamble of A is collision-free
        for idx in range(0, max(self.detect_chirp_offset - 12, 0)):
            self.rec_indexes[0].append(self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])])
            self.rec_bins[0].append(np.max(self.pld_bins[0][idx]))

        # preamble of A is overlapped by preamble or SFD of B
        for idx in range(max(self.detect_chirp_offset - 12, 0), min(self.pld_chirps, self.detect_chirp_offset)):
            curr_index_0 = []
            curr_max_0 = []

            # excluding preamble chirp index of B
            for jdx in range(len(self.pld_indexes[0][idx])):
                if abs(self.pld_indexes[0][idx][jdx] - preamble_index) >= self.pld_tolerance \
                        and self.pld_bins[0][idx][jdx] > self.bin_threshold:
                    curr_index_0.append(self.pld_indexes[0][idx][jdx])
                    curr_max_0.append(self.pld_bins[0][idx][jdx])

            if len(curr_index_0) >= 1:
                self.rec_indexes[0].append(curr_index_0[np.argmax(curr_max_0)])
                self.rec_bins[0].append(np.max(curr_max_0))
            else:
                self.rec_indexes[0].append(self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])])
                self.rec_bins[0].append(np.max(self.pld_bins[0][idx]))

        # chirp of A is partly overlapped by the first payload chirp of B
        if self.detect_chirp_offset < self.pld_chirps:
            self.rec_indexes[0].append(self.pld_indexes[0][self.detect_chirp_offset])
            self.rec_bins[0].append(self.pld_bins[0][self.detect_chirp_offset])

        # payload chirp of A is overlapped by two payload chirps of B
        for idx in range(min(self.pld_chirps, self.detect_chirp_offset + 1), self.pld_chirps):
            # using the first or second segment of A as reference
            if self.detect_samp_offset >= (self.d_samples_per_symbol - self.detect_samp_offset):
                curr_idx = self.pld_indexes[0][idx][1]
                # curr_idx = [[jdx] for jdx in curr_idx]
                curr_max = self.pld_bins[0][idx][1]
            else:
                curr_idx = self.pld_indexes[0][idx][2]
                # curr_idx = [[jdx] for jdx in curr_idx]
                curr_max = self.pld_bins[0][idx][2]
            # counting the number of repeated index
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[0][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[0][idx][jdx][kdx]) <= self.pld_tolerance and self.pld_bins[0][idx][jdx][kdx] > 0:
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

        # payload chirp of B is overlapped by two payload chirps of A
        for idx in range(0, max(0,self. pld_chirps - self.detect_chirp_offset - 1)):
            # using the first or second segment of B as reference
            if self.detect_samp_offset >= (self.d_samples_per_symbol - self.detect_samp_offset):
                curr_idx = self.pld_indexes[1][idx][2]
                # curr_idx = [[jdx] for jdx in curr_idx]
                curr_max = self.pld_bins[1][idx][2]
            else:
                curr_idx = self.pld_indexes[1][idx][1]
                # curr_idx = [[jdx] for jdx in curr_idx]
                curr_max = self.pld_bins[1][idx][1]
            # counting the number of repeated index
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            # curr_idx[ldx].append(pld_indexes[1][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt == 3)[0]

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
            self.rec_indexes[1].append(self.pld_indexes[1][self.pld_chirps - self.detect_chirp_offset - 1])
            self.rec_bins[1].append(self.pld_bins[1][self.pld_chirps - self.detect_chirp_offset - 1])

        # payload of B is collision free
        for idx in range(max(self.pld_chirps - self.detect_chirp_offset, 0), self.pld_chirps):
            self.rec_indexes[1].append(self.pld_indexes[1][idx][np.argmax(self.pld_bins[1][idx])])
            self.rec_bins[1].append(np.max(self.pld_bins[1][idx]))

        # symbol recovery based on repeated index is failed for A
        for idx in range(self.detect_chirp_offset + 1, self.pld_chirps):
            if not (type(self.rec_indexes[0][idx]) is int):
                curr_idx = []
                oly_idx = []
                # payload chirp index of B when aligning with chirp of A
                for jdx in range(idx - self.detect_chirp_offset - 1, idx - self.detect_chirp_offset + 1):
                    if type(self.rec_indexes[1][jdx]) is int:
                        oly_idx.append((self.rec_indexes[1][jdx] - self.detect_bin_offset + self.d_number_of_bins) % self.d_number_of_bins)

                # excluding payload chirp index of B
                for jdx in range(len(self.rec_indexes[0][idx])):
                    is_oly = False
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[0][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[0][idx][jdx])

                if len(curr_idx) == 1:
                    # cross decoding successes
                    self.rec_indexes[0][idx] = curr_idx[0]
                else:
                    # cross decoding is also failed
                    # performing power mapping
                    if self.detect_samp_offset >= (self.d_samples_per_symbol - self.detect_samp_offset):
                        jdx = 1
                    else:
                        jdx = 2
                    # utilizing the first or second segment of A as reference
                    if self.sfd_bins[0] >= self.sfd_bins[1]:
                        self.rec_indexes[0][idx] = self.pld_indexes[0][idx][jdx][0]
                        self.rec_bins[0][idx] = self.pld_bins[0][idx][jdx][0]
                    else:
                        self.rec_indexes[0][idx] = self.pld_indexes[0][idx][jdx][1]
                        self.rec_bins[0][idx] = self.pld_bins[0][idx][jdx][1]

        # symbol recovery based on repeated index is failed for B
        for idx in range(0, self.pld_chirps - self.detect_chirp_offset - 1):
            if not (type(self.rec_indexes[1][idx]) is int):
                curr_idx = []
                oly_idx = []
                # payload chirp index of B when aligning with chirp of A
                for jdx in range(idx + self.detect_chirp_offset, idx + self.detect_chirp_offset + 2):
                    if type(self.rec_indexes[0][jdx]) is int:
                        oly_idx.append((self.rec_indexes[0][jdx] + self.detect_bin_offset + self.d_number_of_bins) % self.d_number_of_bins)

                # excluding payload chirp index of A
                for jdx in range(len(self.rec_indexes[1][idx])):
                    is_oly = False
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[1][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[1][idx][jdx])

                if len(curr_idx) == 1:
                    # cross decoding successes
                    self.rec_indexes[1][idx] = curr_idx[0]
                else:
                    # cross decoding is also failed
                    # performing power mapping
                    if self.detect_samp_offset <= (self.d_samples_per_symbol - self.detect_samp_offset):
                        jdx = 1
                    else:
                        jdx = 2
                    # utilizing the first or second segment of A as reference
                    if self.sfd_bins[0] <= self.sfd_bins[1]:
                        self.rec_indexes[1][idx] = self.pld_indexes[1][idx][jdx][0]
                        self.rec_bins[1][idx] = self.pld_bins[1][idx][jdx][0]
                    else:
                        self.rec_indexes[1][idx] = self.pld_indexes[1][idx][jdx][1]
                        self.rec_bins[1][idx] = self.pld_bins[1][idx][jdx][1]

        if self.detect_chirp_offset < self.pld_chirps:
            idx = self.detect_chirp_offset
            if type(self.rec_indexes[1][0]) is int:
                # the first payload chirp index of B when aligning chirp of A
                oly_idx = [(self.rec_indexes[1][0] - self.detect_bin_offset + self.d_number_of_bins) % self.d_number_of_bins]
                curr_idx = []
                curr_max = []
                # excluding the first payload chirp index of B
                for jdx in range(len(self.pld_indexes[0][idx])):
                    is_oly = False
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[0][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[0][idx][jdx])
                        curr_max.append(self.rec_bins[0][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[0][idx] = curr_idx[0]
                    self.rec_bins[0][idx] = curr_max[0]
                else:
                    self.rec_indexes[0][idx] = self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])]
                    self.rec_bins[0][idx] = np.max(self.pld_bins[0][idx])
            else:
                self.rec_indexes[0][idx] = self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])]
                self.rec_bins[0][idx] = np.max(self.pld_bins[0][idx])

            idx = self.pld_chirps - self.detect_chirp_offset - 1
            if type(self.rec_indexes[0][self.pld_chirps - 1]) is int:
                # the last payload chirp index of A when aligning chirp of B
                oly_idx = [(self.rec_indexes[0][self.pld_chirps - 1] + self.detect_bin_offset + self.d_number_of_bins) % self.d_number_of_bins]
                curr_idx = []
                curr_max = []
                # excluding the last payload chirp index of A
                for jdx in range(len(self.pld_indexes[1][idx])):
                    is_oly = False
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[1][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[1][idx][jdx])
                        curr_max.append(self.rec_bins[1][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[1][idx] = curr_idx[0]
                    self.rec_bins[1][idx] = curr_max[0]
                else:
                    self.rec_indexes[1][idx] = self.pld_indexes[1][idx][np.argmax(self.pld_bins[1][idx])]
                    self.rec_bins[1][idx] = np.max(self.pld_bins[1][idx])
            else:
                self.rec_indexes[1][idx] = self.pld_indexes[1][idx][np.argmax(self.pld_bins[1][idx])]
                self.rec_bins[1][idx] = np.max(self.pld_bins[1][idx])

        # recording the payload chirp index of B when aligning with A
        self.oly_indexes.append(np.array(
            np.mod(np.add(self.rec_indexes[1], self.d_number_of_bins - self.detect_bin_offset), self.d_number_of_bins), dtype=np.int))
        # recording the payload chirp index of A when aligning with B
        self.oly_indexes.append(np.array(
            np.mod(np.add(self.rec_indexes[0], self.d_number_of_bins + self.detect_bin_offset), self.d_number_of_bins), dtype=np.int))


    def error_correct(self):
        index_error = np.array([0, 0])
        for idx in range(8):
            ogn_index = [self.rec_indexes[0][idx], self.rec_indexes[1][idx]]
            curr_index = np.int16(np.rint(np.divide(ogn_index, 4)))
            index_error = np.add(index_error, np.subtract(np.multiply(curr_index, 4), ogn_index))
            self.corr_indexes[0].append(4*curr_index[0])
            self.corr_indexes[1].append(4*curr_index[1])

        index_error = np.rint(np.divide(index_error, 8))
        for idx in range(8, self.pld_chirps):
            ogn_index = [self.rec_indexes[0][idx], self.rec_indexes[1][idx]]
            curr_index = np.int16(np.mod(np.add(np.add(ogn_index, index_error), self.d_number_of_bins), self.d_number_of_bins))
            self.corr_indexes[0].append(curr_index[0])
            self.corr_indexes[1].append(curr_index[1])

    def get_chirp_error(self):
        chirp_errors_list = np.abs(np.subtract(self.packet_chirp[self.packet_index], self.corr_indexes))
        self.chirp_errors.append(np.count_nonzero(chirp_errors_list[0]))
        self.chirp_errors.append(np.count_nonzero(chirp_errors_list[1]))

    def recover_byte(self):
        self.rec_bytes.append(self.lora_decoder.decode(self.corr_indexes[0].copy()))
        # if self.d_debug >= 2:
        #     print(self.rec_bytes)
        self.rec_bytes.append(self.lora_decoder.decode(self.corr_indexes[1].copy()))
        # if self.d_debug >= 2:
        #     print(self.rec_bytes)

    def get_byte_error(self):

        for idx in range(2):
            comp_result = []
            header_result = []
            for jdx in range(0, 3):
                header_result.append(bin(self.packet_byte[self.packet_index][idx][jdx] ^ self.rec_bytes[idx][jdx]).count('1'))
            for jdx in range(0, self.packet_length):
                comp_result.append(bin(self.packet_byte[self.packet_index][idx][jdx] ^ self.rec_bytes[idx][jdx]).count('1'))

            self.byte_errors.append(np.sum(comp_result))
            self.header_errors.append(np.sum(header_result))

    def save_result(self):
        fail = True
        if len(self.sfd_begins) == 2:
            if abs(self.sfd_begins[1] - self.sfd_begins[0] - self.time_offset) < self.sfd_tolerance * self.d_decim_factor:
                fail = False

                curr_offsets = [self.chirp_offset, self.samp_offset, self.time_offset]
                for idx in range(3):
                    self.detected_offsets[idx].append(curr_offsets[idx])

                curr_prebins = self.preamble_bins.copy()
                curr_sfdbins = self.sfd_bins.copy()
                curr_chirps = self.corr_indexes.copy()
                curr_bytes = self.rec_bytes.copy()
                curr_chirp_errors = self.chirp_errors.copy()
                curr_header_errors = self.header_errors.copy()
                curr_byte_errors = self.byte_errors.copy()

                for idx in range(2):
                    self.decoded_prebins[idx].append(curr_prebins[idx])
                    self.decoded_sfdbins[idx].append(curr_sfdbins[idx])
                    self.decoded_chirps[idx].append(curr_chirps[idx])
                    self.decoded_bytes[idx].append(curr_bytes[idx])
                    self.decoded_chirp_errors[idx].append(curr_chirp_errors[idx])
                    self.decoded_header_errors[idx].append(curr_header_errors[idx])
                    self.decoded_byte_errors[idx].append(curr_byte_errors[idx])

                if self.d_debug >= 2:
                    self.show_result()
        if fail:
            if self.d_debug>=0:
                self.show_failed()


    def clear_result(self):
        # clear all saved results
        for idx in range(3):
            self.detected_offsets[idx].clear()

        for idx in range(2):
            self.decoded_prebins[idx].clear()
            self.decoded_sfdbins[idx].clear()
            self.decoded_chirps[idx].clear()
            self.decoded_bytes[idx].clear()
            self.decoded_chirp_errors[idx].clear()
            self.decoded_header_errors[idx].clear()
            self.decoded_byte_errors[idx].clear()

    def show_info(self):
        print("Show Info:")
        print("Packet Order: ", self.packet_idx)
        print("Chirp Offset: ", self.chirp_offset)
        print("Samp Offset: ", self.samp_offset)
        print("Time Offset: ", self.time_offset)
        print()

    def show_result(self):
        print("Show      result:")
        print("Detect Chirp Offset: ", self.detect_chirp_offset)
        print("Detect Samp Offset: ", self.detect_samp_offset)
        print("Detect Time Offset: ", self.detect_offset)

        print("Preamble  successful!")
        print("Preamble  indexes: ", self.preamble_indexes)
        print("Preamble  bins   : ", self.preamble_bins)
        print("SFD       indexes: ", self.sfd_indexes)
        print("SFD       bins   : ", self.sfd_bins)
        print("SFD       begins : ", self.sfd_begins)
        print("Downchirp successful!")
        print("PLD       indexe0: ", self.pld_indexes[0])
        print("PLD       indexe1: ", self.pld_indexes[1])
        print("PLD       bins0  : ", self.pld_bins[0])
        print("PLD       bins1  : ", self.pld_bins[1])
        print("REC       indexe0: ", self.rec_indexes[0])
        print("REC       indexe1: ", self.rec_indexes[1])
        print("REC       bins0  : ", self.rec_bins[0])
        print("REC       bins1  : ", self.rec_bins[1])
        print("CORR      indexe0: ", self.corr_indexes[0])
        print("Packet     chirp0: ", self.packet_chirp[self.packet_index][0])
        print("CORR      indexe1: ", self.corr_indexes[1])
        print("Packet     chirp1: ", self.packet_chirp[self.packet_index][1])
        print("Chirp      Errors: ", self.chirp_errors)
        print("Header     Errors: ", self.header_errors)
        print("Byte       Errors: ", self.byte_errors)

    def show_failed(self):
        print("Sync      failed!")
        print("Detect Chirp Offset: ", self.detect_chirp_offset)
        print("Detect Samp Offset: ", self.detect_samp_offset)
        print("Detect Time Offset: ", self.detect_offset)
        print("Preamble  indexes: ", self.preamble_indexes)
        print("Preamble  bins   : ", self.preamble_bins)
        print("SFD       indexes: ", self.sfd_indexes)
        print("SFD       bins   : ", self.sfd_bins)
        print("SFD       begins : ", self.sfd_begins)

    def show_chirp(self, bgn_idx, end_idx, packet_index):
        chirp_ogn = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

        chirp_oly = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
        offset = self.time_offset % self.d_samples_per_symbol

        if packet_index == 0:
            chirp_ogn[0:len(self.packet_shr[0][bgn_idx:end_idx])] = self.packet_shr[0][bgn_idx:end_idx]
            if (end_idx - self.time_offset) > 0:
                if (end_idx - self.time_offset) < self.d_samples_per_symbol:
                    chirp_oly[offset:] = \
                        self.packet_shr[1][bgn_idx - self.time_offset + offset:end_idx - self.time_offset]
                else:
                    chirp_oly = self.packet_shr[1][bgn_idx - self.time_offset:end_idx - self.time_offset]
        else:
            chirp_ogn[0:len(self.packet_shr[1][bgn_idx - self.time_offset:end_idx - self.time_offset])] = \
                self.packet_shr[1][bgn_idx - self.time_offset:end_idx - self.time_offset]
            if bgn_idx < len(self.packet_shr[0]):
                if (len(self.packet_shr[0]) - bgn_idx) < self.d_samples_per_symbol:
                    chirp_oly[0:self.d_samples_per_symbol - offset] = \
                        self.packet_shr[0][bgn_idx:end_idx - offset]
                else:
                    chirp_oly[0:len(self.packet_shr[0][bgn_idx:end_idx])] = self.packet_shr[0][bgn_idx:end_idx]

        chirp_ogn_index, chirp_ogn_max, chirp_ogn_bin = \
            self.get_fft_bins(chirp_ogn, self.d_samples_per_symbol, self.detect_scope, self.preamble_peak)

        chirp_oly_index, chirp_oly_max, chirp_oly_bin = \
            self.get_fft_bins(chirp_oly, self.d_samples_per_symbol, self.detect_scope, self.preamble_peak)

        chirp_ogn_ifreq = self.get_chirp_ifreq(chirp_ogn)
        chirp_oly_ifreq = self.get_chirp_ifreq(chirp_oly)

        self.show_signal([chirp_ogn_bin, chirp_oly_bin], 'Ogn FFT 0,1')
        self.show_signal([chirp_ogn_ifreq, chirp_oly_ifreq], 'Ogn ifreq 0,1')

    @staticmethod
    def get_chirp_ifreq(content):
        curr_conj_mul = []
        for idx in range(1, len(content)):
            curr_conj_mul.append(content[idx] * np.conj(content[idx - 1]))
        curr_ifreq = [phase(conj_mul) for conj_mul in curr_conj_mul]
        curr_ifreq.append(curr_ifreq[-1])
        return curr_ifreq

    @staticmethod
    def show_signal(content, title):
        x = [idx for idx in range(len(content[0]))]
        plt.plot(x, content[0], 'k.--')
        plt.plot(x, content[1], 'r.--')
        plt.title(title)
        plt.show()

def save_as_object(obj, name, sf, length, power):
    path_prefix = '../offResult/mlora2/'

    file_name = path_prefix + name +'_sf_' + str(sf) + '_len_' + str(length[0]) + '_pow_' + str(power[0]) + '_' + str(power[1]) + '.pkl'
    with open(file_name, 'wb') as output:
        pl.dump(obj, output, pl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    test_bw = 125e3
    test_samples_per_second = 1e6
    test_sf = 7
    test_cr = 4
    test_powers = [-3, 0]
    test_lengths = [16, 16]

    offresult2=OffResult()

    mlora2 = MLoRa2(test_bw, test_samples_per_second, test_sf, test_cr, test_powers, test_lengths)
    mlora2.read_packets()
    mlora2.build_ideal_chirp()

    lora2 = LoRa2(test_bw, test_samples_per_second, test_sf, test_cr, test_powers, test_lengths)
    lora2.read_packets()
    lora2.build_ideal_chirp()

    for i in range(10):
        for j in range(100):
            mlora2.read_packet_shr(j)
            mlora2.init_decoding()
            mlora2.decoding_packet()

            lora2.read_packet_shr(j, mlora2.chirp_offset, mlora2.samp_offset, mlora2.packet_o)
            lora2.init_decoding()
            lora2.decoding_packet()

        offresult2.mlora_offsets.append([mlora2.detected_offsets[j].copy() for j in range(3)])
        offresult2.mlora_prebins.append([mlora2.decoded_prebins[j].copy() for j in range(2)])
        offresult2.mlora_sfdbins.append([mlora2.decoded_sfdbins[j].copy() for j in range(2)])
        offresult2.mlora_chirps.append([mlora2.decoded_chirps[j].copy() for j in range(2)])
        offresult2.mlora_bytes.append([mlora2.decoded_bytes[j].copy() for j in range(2)])
        offresult2.mlora_chirp_errors.append([mlora2.decoded_chirp_errors[j].copy() for j in range(2)])
        offresult2.mlora_header_errors.append([mlora2.decoded_header_errors[j].copy() for j in range(2)])
        offresult2.mlora_byte_errors.append([mlora2.decoded_byte_errors[j].copy() for j in range(2)])

        offresult2.lora_prebins.append(lora2.decoded_prebins.copy())
        offresult2.lora_sfdbins.append(lora2.decoded_sfdbins.copy())
        offresult2.lora_chirps.append(lora2.decoded_chirps.copy())
        offresult2.lora_bytes.append(lora2.decoded_bytes.copy())
        offresult2.lora_chirp_errors.append(lora2.decoded_chirp_errors.copy())
        offresult2.lora_header_errors.append(lora2.decoded_header_errors.copy())
        offresult2.lora_byte_errors.append(lora2.decoded_byte_errors.copy())

        mlora2.clear_result()
        lora2.clear_result()

    save_as_object(offresult2, 'offresult2', test_sf, test_lengths, test_powers)
    # save_as_object(mlora2, 'mlora2', test_sf, test_lengths, test_powers)
