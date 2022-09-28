import pickle as pl
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from cmath import phase
from scipy.fftpack import fft
from Stack import Stack, State
from lora_decode import LoRaDecode
from LoRa3 import LoRa3
from OffResult import OffResult

debug_off = -1
debug_info = 1
debug_verbose = 2
debug_on = debug_off

class MLoRa3:

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
        self.preamble_peak = 5
        self.bin_threshold = 9
        self.preamble_chirps = 7
        self.preamble_tolerance = 2
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_history = []
        self.prebin_history = []
        for idx in range(5):
            self.pre_history.append(Stack())
            self.prebin_history.append(Stack())

        self.pre_ordered = False
        self.pre_sfd_dist = 4
        self.sfd_dist = [0, 0, 0]
        self.sfd_chirps = 2
        self.sfd_peak = 5
        self.sfd_threshold = 5
        self.sfd_tolerance = 4
        self.sfd_pre_tol = 1 # sfd and preamble tolerance:检测preamble index时，需要排除前一个进入sfd state的index，这里是他们的容忍误差
        self.detect_offset_tol = 8 # 检测得的offset tolerance
        self.sfd_indexes = []
        self.sfd_bins = []
        self.sfd_begins = []
        self.sfd_history = []
        self.sfdbin_history = []
        self.sfdbins_history = []
        self.sync_fail_cnt = 0
        self.sync_suc_cnt = 0

        self.pld_peak = 3
        self.detect_offset = None
        self.detect_chirp_offset = None
        self.detect_samp_offset = None
        self.detect_bin_offset = None
        self.shift_samples = []
        self.pld_chirps = int(8 + (4 + self.d_cr) * np.ceil(self.d_lengths[0] * 2.0 / self.d_sf))
        self.pld_tolerance = 2
        self.pld_begins = []
        self.pld_indexes = [[], [], []]
        self.pld_bins = [[], [], []]

        self.lora_decoder = LoRaDecode(self.payload_length, self.d_sf, self.d_cr)
        self.rec_indexes = [[], [], []]
        self.rec_bins = [[], [], []]
        self.rec_results = [[], [], []]
        self.oly_indexes = []
        self.corr_indexes = [[], [], []]
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_states = [State.S_RESET, State.S_RESET, State.S_RESET]

        # Save result
        self.chirp_errors_list = None
        self.detected_offsets = []
        self.decoded_prebins = [[], [], []]
        self.decoded_sfdbins = [[], [], []]
        self.decoded_chirps = [[], [], []]
        self.decoded_bytes = [[], [], []]
        self.decoded_chirp_errors = [[], [], []]
        self.decoded_header_errors = [[], [], []]
        self.decoded_byte_errors = [[], [], []]

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

    def read_packet_shr(self, packet_idx):
        # all time offset space
        # self.chirp_offset = np.random.randint(0, self.chirp_cnt)
        # self.samp_offset = np.random.randint(0, self.d_samples_per_symbol)
        # self.time_offset = self.chirp_offset * self.d_samples_per_symbol + self.samp_offset

        # controlled time offset space
        self.chirp_cnt = len(self.packets_shr[0][0]) // self.d_samples_per_symbol
        self.detect_scope = 2

        # generate random chirp offset AB < AC
        chirp_offset1 = np.random.randint(0, self.chirp_cnt - 1)
        chirp_offset2 = np.random.randint(chirp_offset1, self.chirp_cnt-1)
        self.chirp_offset = np.array([chirp_offset1, chirp_offset2])

        # generate random samp offset AB < AC
        interval = 8
        samp_offset1 = np.random.randint(interval, self.d_number_of_bins - 2*interval)
        samp_offset2 = np.random.randint(samp_offset1 + interval, self.d_number_of_bins - interval)
        self.samp_offset = np.multiply(np.array([samp_offset1, samp_offset2]), self.d_decim_factor)

        # configured time offset
        # self.chirp_offset = np.array([9, 24])
        # self.samp_offset = np.array([344, 592])

        # self.samp_offset = np.multiply(np.array([69, 106]), self.d_decim_factor)

        self.time_offset = np.add(np.multiply(self.chirp_offset, self.d_samples_per_symbol), self.samp_offset)

        self.packet_idx = packet_idx
        self.packet = []
        self.packet_shr = []
        for idx in range(self.packet_nums):
            self.packet.append(self.packets[idx][packet_idx])
            self.packet_shr.append(self.packets_shr[idx][packet_idx])

        # Only A
        self.packet_o = self.packet_shr[0][:self.time_offset[0]]

        # Overlap of A and B
        packet_o_ab = np.add(self.packet_shr[0][self.time_offset[0]:self.time_offset[1]], self.packet_shr[1][:self.time_offset[1] - self.time_offset[0]])
        self.packet_o = np.concatenate((self.packet_o, packet_o_ab))

        # Overlap of A, B and C
        packet_o_abc = np.add(self.packet_shr[0][self.time_offset[1]:], self.packet_shr[1][self.time_offset[1] - self.time_offset[0]:len(self.packet_shr[0])-self.time_offset[0]])
        packet_o_abc = np.add(packet_o_abc, self.packet_shr[2][:len(self.packet_shr[0])-self.time_offset[1]])
        self.packet_o = np.concatenate((self.packet_o, packet_o_abc))

        # Overlap of B and C
        packet_o_bc = np.add(self.packet_shr[1][len(self.packet_shr[0])-self.time_offset[0]:], self.packet_shr[2][len(self.packet_shr[0])-self.time_offset[1]:len(self.packet_shr[1])-self.time_offset[1]+self.time_offset[0]])
        self.packet_o = np.concatenate((self.packet_o, packet_o_bc))

        # Only C
        self.packet_o = np.concatenate((self.packet_o, self.packet_shr[2][len(self.packet_shr[1])-self.time_offset[1]+self.time_offset[0]:]))

        if self.d_debug >= 1:
            self.show_info()


    def init_decoding(self):
        self.bin_threshold = 9
        self.preamble_chirps = 7
        self.preamble_tolerance = 2
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_history = []
        self.prebin_history = []
        for idx in range(5):
            self.pre_history.append(Stack())
            self.prebin_history.append(Stack())

        self.pre_ordered = False
        self.pre_sfd_dist = 4
        self.sfd_dist = [0, 0]
        self.sfd_chirps = 2
        self.sfd_threshold = 5
        self.sfd_tolerance = 4
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
        self.pld_begins = []
        self.pld_indexes = [[], [], []]
        self.pld_bins = [[], [], []]

        self.rec_indexes = [[], [], []]
        self.rec_bins = [[], [], []]
        self.rec_results = [[], [], []]
        self.oly_indexes = []
        self.corr_indexes = [[], [], []]
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_states = [State.S_RESET, State.S_RESET, State.S_RESET]
        for idx in range(len(self.d_states)):
            if self.d_states[idx] == State.S_RESET:
                for jdx in range(5):
                    self.pre_history[jdx].clear()
                    self.prebin_history[jdx].clear()
                self.d_states[idx] = State.S_PREFILL

    # 检测preamble位置
    def decoding_packet(self):
        if self.d_debug >= 1:
            print("---------------------------")
            print(" Decoding Packet")

        idx = 0
        while idx < len(self.packet_o):
            if idx + self.d_samples_per_symbol > len(self.packet_o):
                break

            bgn_index = idx
            end_index = idx + self.d_samples_per_symbol
            chirp_o = self.packet_o[bgn_index:end_index]
            chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp_o, self.d_samples_per_symbol, self.detect_scope, self.preamble_peak)

            for jdx in range(5):
                self.pre_history[jdx].push(chirp_index[jdx])
                self.prebin_history[jdx].push(chirp_max[jdx])

                if self.pre_history[jdx].size() > self.preamble_chirps:  # preamble chirps here is a prefill threshold
                    self.pre_history[jdx].pop_back()
                    self.prebin_history[jdx].pop_back()

            if self.d_states[0] == State.S_PREFILL and self.d_states[1] == State.S_PREFILL \
                    and self.d_states[2] == State.S_PREFILL:
                # 假如有一个超过了preamble的长度，那么三个都进入 DETECT_PREAMBLE 阶段。
                if self.pre_history[0].size() >= self.preamble_chirps:
                    for jdx in range(len(self.d_states)):
                        self.d_states[jdx] = State.S_DETECT_PREAMBLE
                else:
                    idx += self.d_samples_per_symbol
                    continue

            if self.d_states[0] == State.S_DETECT_PREAMBLE or self.d_states[1] == State.S_DETECT_PREAMBLE or \
                    self.d_states[2] == State.S_DETECT_PREAMBLE:
                # 对已经存入的preamble进行分析，全匹配，找到连续6个index为0的片段，则认为是preamble开始了
                # detect preamble会返回[0 96 -1]这种，表示A中有index为0，B中index 为96
                detect_indexes, detect_bins = self.detect_preamble_all(self.pre_history, self.prebin_history)
                # detect_indexes, detect_bins = self.detect_preamble(self.pre_history, self.prebin_history)

                # case 1: A B C 都在 preamble
                if self.d_states[0] == State.S_DETECT_PREAMBLE and self.d_states[1] == State.S_DETECT_PREAMBLE and \
                        self.d_states[2] == State.S_DETECT_PREAMBLE:
                    if len(np.where(detect_indexes != -1)[0]) == 0:
                        # 似乎这里并不会执行，因为A包的offset设为0，总是会检测到包的
                        idx += self.d_samples_per_symbol
                        continue
                    else:
                        # 可以理解为就是preamble index
                        repeat_index = np.where(detect_indexes != -1)[0]
                        if len(repeat_index) >= 3:
                            # 大于3说明ABC都检测到了preamble
                            for jdx in range(3):
                                self.d_states[jdx] = State.S_SFD_SYNC
                                self.preamble_indexes.append(detect_indexes[repeat_index[jdx]])
                                self.preamble_bins.append(detect_bins[repeat_index[jdx]])

                            # align with the first packet
                            # i -= preamble_indexes[0] * d_decim_factor
                            # preamble_indexes[1] += preamble_indexes[0]
                        elif len(repeat_index) == 2:
                            # 大于2说明AB都检测到了preamble
                            for jdx in range(2):
                                self.d_states[jdx] = State.S_SFD_SYNC
                                self.preamble_indexes.append(detect_indexes[repeat_index[jdx]])
                                self.preamble_bins.append(detect_bins[repeat_index[jdx]])
                        elif len(repeat_index) == 1:
                            # 否则检测出来了一个，即A
                            self.d_states[0] = State.S_SFD_SYNC
                            self.preamble_indexes.append(detect_indexes[repeat_index[0]])
                            self.preamble_bins.append(detect_bins[repeat_index[0]])
                            # align with the first packet
                            # i -= preamble_indexes[0] * d_decim_factor

                        idx += self.d_samples_per_symbol
                        continue

                # case 2: B C 同在preamble
                # [0]的preamble已经被检测出来了，进入了S_SFD_SYNC，另一个还在PREAMBLE
                if self.d_states[0] != State.S_DETECT_PREAMBLE and self.d_states[1] == State.S_DETECT_PREAMBLE and \
                        self.d_states[2] == State.S_DETECT_PREAMBLE:
                    # A 在SYNC
                    if self.d_states[0] == State.S_SFD_SYNC:
                        unique_index_b = \
                            np.where((detect_indexes != -1) & (detect_indexes - self.preamble_indexes[0] > self.sfd_pre_tol))[0]

                        if len(unique_index_b) > 0 and self.d_states[0] == State.S_SFD_SYNC:
                            self.d_states[1] = State.S_SFD_SYNC
                            self.preamble_indexes.append(detect_indexes[unique_index_b[0]])
                            self.preamble_bins.append(detect_bins[unique_index_b[0]])

                        if len(unique_index_b) > 1:
                            unique_index_c = \
                                np.where((detect_indexes != -1) & (detect_indexes - self.preamble_indexes[0] > self.sfd_pre_tol) &
                                         (detect_indexes - unique_index_b[1] > self.sfd_pre_tol))[0]
                            if len(unique_index_c) > 0:
                                self.preamble_indexes.append(detect_indexes[unique_index_c[0]])
                                self.preamble_bins.append(detect_bins[unique_index_c[0]])
                                self.d_states[2] = State.S_SFD_SYNC
                    # A 在payload
                    elif self.d_states[0] == State.S_READ_PAYLOAD:
                        repeat_index = np.where(detect_indexes != -1)[0]
                        if len(repeat_index) == 1:
                            self.d_states[1] = State.S_SFD_SYNC
                            self.preamble_indexes.append(detect_indexes[repeat_index[0]])
                            self.preamble_bins.append(detect_bins[repeat_index[0]])
                        elif len(repeat_index) == 2:
                            for jdx in range(1, 3):
                                self.d_states[jdx] = State.S_SFD_SYNC
                                self.preamble_indexes.append(detect_indexes[repeat_index[jdx - 1]])
                                self.preamble_bins.append(detect_bins[repeat_index[jdx - 1]])

                # case 3: C 在 preamble
                if self.d_states[0] != State.S_DETECT_PREAMBLE and self.d_states[1] != State.S_DETECT_PREAMBLE and \
                        self.d_states[2] == State.S_DETECT_PREAMBLE:
                    unique_index_c = np.where((detect_indexes != -1) & (detect_indexes - self.preamble_indexes[0] > self.sfd_pre_tol) &
                                              (detect_indexes - self.preamble_indexes[1]) > self.sfd_pre_tol)[0]
                    repeat_index = np.where(detect_indexes != -1)[0]
                    # B 不在payload，可能仍然在检测upchirp
                    if len(unique_index_c) > 0 and (self.d_states[1] == State.S_SFD_SYNC
                                                    or self.d_states[0] == State.S_SFD_SYNC):
                        self.d_states[2] = State.S_SFD_SYNC
                        self.preamble_indexes.append(detect_indexes[unique_index_c[0]])
                        self.preamble_bins.append(detect_bins[unique_index_c[0]])

                    # B 在payload， C直接取就好了
                    if len(repeat_index) > 0 and self.d_states[0] != State.S_SFD_SYNC and \
                            self.d_states[1] != State.S_SFD_SYNC:
                        self.d_states[2] = State.S_SFD_SYNC
                        self.preamble_indexes.append(detect_indexes[repeat_index[0]])
                        self.preamble_bins.append(detect_bins[repeat_index[0]])

            # 有一个已经到了SFD start of ，则开始检测 downchirp 了
            if self.d_states[0] == State.S_SFD_SYNC or self.d_states[1] == State.S_SFD_SYNC or \
                    self.d_states[2] == State.S_SFD_SYNC:
                #
                # 可以穷举6种情况, ABC 100 010 001 | 110 011 111
                # 100
                if self.d_states[0] == State.S_SFD_SYNC:
                    # 计算downchirp的起始位置，修正对齐
                    bgn_sfd = idx - self.preamble_indexes[0] * self.d_decim_factor
                    if self.detect_down_chirp(bgn_sfd, self.detect_scope, self.sfd_peak):
                        self.d_states[0] = State.S_READ_PAYLOAD
                        # i += 2*d_samples_per_symbol + d_quad_samples_per_symbol - preamble_indexes[0] * d_decim_factor
                        # Record shift samples needed to align with the first packet
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                                  - self.preamble_indexes[0] * self.d_decim_factor)
                        self.sfd_begins.append(bgn_sfd)
                # 010
                if self.d_states[1] == State.S_SFD_SYNC:
                    bgn_sfd = idx - self.preamble_indexes[1] * self.d_decim_factor
                    if self.detect_down_chirp(bgn_sfd, self.detect_scope, self.sfd_peak):
                        self.d_states[1] = State.S_READ_PAYLOAD
                        # Record shift samples need to align with the second packet
                        # 记录偏移量
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                                  - self.preamble_indexes[1] * self.d_decim_factor)
                        self.sfd_begins.append(bgn_sfd)
                # 001
                if self.d_states[2] == State.S_SFD_SYNC:
                    bgn_sfd = idx - self.preamble_indexes[2] * self.d_decim_factor
                    if self.detect_down_chirp(bgn_sfd, self.detect_scope, self.sfd_peak):
                        self.d_states[2] = State.S_READ_PAYLOAD
                        # Record shift samples need to align with the second packet
                        # 记录偏移量
                        self.shift_samples.append(2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
                                                  - self.preamble_indexes[2] * self.d_decim_factor)
                        self.sfd_begins.append(bgn_sfd)
            # if self.d_states[2] == State.S_SFD_SYNC:
            #     print(self.sfd_begins)
            # if self.d_states[2] == State.S_READ_PAYLOAD:
            #     print(self.sfd_begins)
            idx += self.d_samples_per_symbol
            continue

        # Stop Detection
        if len(self.sfd_begins) < 3:
            self.sync_fail_cnt += 1
            if self.d_debug >= 1:
                print("NOT DETECT 3 sfd bins!!!")
            return

        # self.detect_chirp_offset_ab = (self.sfd_begins[1] - self.sfd_begins[0]) // self.d_samples_per_symbol
        # self.detect_chirp_offset_ac = (self.sfd_begins[2] - self.sfd_begins[0]) // self.d_samples_per_symbol
        # self.detect_samp_offset_ab = (self.sfd_begins[1] - self.sfd_begins[0]) - \
        #                              self.detect_chirp_offset_ab * self.d_samples_per_symbol
        # self.detect_samp_offset_ac = (self.sfd_begins[2] - self.sfd_begins[0]) - \
        #                              self.detect_chirp_offset_ac * self.d_samples_per_symbol
        # self.detect_offset_ab = self.sfd_begins[1] - self.sfd_begins[0]
        # self.detect_offset_ac = self.sfd_begins[2] - self.sfd_begins[0]
        #
        # if self.d_debug>=2:
        #     print("CORRECT TIME OFFSET:", self.time_offset[0], self.time_offset[1])
        #     print("CORRECT BIN OFFSET", self.samp_offset[0] // self.d_decim_factor,
        #           self.samp_offset[1] // self.d_decim_factor)
        #
        #     print("CHIRP OFF AB: ", self.detect_chirp_offset_ab)
        #     print("CHIRP OFF AC: ", self.detect_chirp_offset_ac)
        #     print("SAMPLE OFF AB: ", self.detect_samp_offset_ab // self.d_decim_factor)
        #     print("SAMPLE OFF AC: ", self.detect_samp_offset_ac // self.d_decim_factor)
        #
        #
        #
        # if abs(self.detect_offset_ab - self.time_offset[0]) > self.detect_offset_tol or \
        #         abs(self.detect_offset_ac - self.time_offset[1]) > self.detect_offset_tol:
        #     self.sync_fail_cnt += 1
        #     if self.d_debug>=1:
        #         print("----------FAIL!")
        # else:
        #     sfd_bgn_index = 10 * self.d_samples_per_symbol
        #     bgn_indexes = [sfd_bgn_index, sfd_bgn_index + self.time_offset[0], sfd_bgn_index + self.time_offset[1]]
        #     self.sfd_begins = bgn_indexes
        #     down_off = 2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
        #     for i in range(len(self.sfd_begins)):
        #         self.pld_begins.append(self.sfd_begins[i] + down_off)
        #
        #     self.sync_suc_cnt += 1
        #     if self.d_debug>=1:
        #         print("SFD     BEGINS:", self.sfd_begins)
        #         print("PAYLOAD BEGINS:", self.pld_begins)
        #         print("-------SUCCEED!")

        self.packet_demode()
        self.packet_decode()


    def packet_demode(self):
        if not self.get_sfd_bin(self.detect_scope, self.sfd_peak):
            self.show_failed()
            return

        # Beginning position of payload according to sfd_begins
        sfd_length = 2*self.d_samples_per_symbol + self.d_quad_samples_per_symbol
        for idx in range(len(self.sfd_begins)):
            self.pld_begins.append(self.sfd_begins[idx]+sfd_length)

        if self.d_debug >= 1:
            print('Decode  Packet : ')
            print('PRE     Indexes: ', self.preamble_indexes)
            print('SFD     Begins : ', self.sfd_begins)
            print('Payload Begins : ', self.pld_begins)

        # Save FFT indexes and bins
        if self.d_debug>=2:
            print('Symbol Recovery A: ')
            print('Free A:            ', [0, max(self.chirp_offset[0]-12, 0)])
            print('A and Pre B        ', [max(self.chirp_offset[0] - 12, 0), max(min(self.chirp_offset[1]-12, self.chirp_offset[0]), 0)])
            print('A , Pre B and Pre C', [max(min(self.chirp_offset[1] - 12, self.chirp_offset[0]), 0), min(self.chirp_offset[0], self.pld_chirps)])
            print('A and B            ', [min(self.chirp_offset[0], self.pld_chirps), min(max(self.chirp_offset[0], self.chirp_offset[1]- 12),self.pld_chirps)])
            print('A , B and Pre C    ', [min(max(self.chirp_offset[0], self.chirp_offset[1] - 12), self.pld_chirps), min(self.chirp_offset[1], self.pld_chirps)])
            print('A , B and C        ', [min(self.chirp_offset[1], self.pld_chirps), self.pld_chirps])

            print('Symbol Recovery B: ')
            print('A and B            ', [0, max(self.chirp_offset[1] - self.chirp_offset[0] - 12, 0)])
            print('A , B and Pre C    ', [max(self.chirp_offset[1]-self.chirp_offset[0]-12, 0), max(min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps-self.chirp_offset[0]), 0)])
            print('A , B and C        ', [max(min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps-self.chirp_offset[0]), 0), max(self.pld_chirps-self.chirp_offset[0], 0)])
            print('B and Pre C        ', [max(self.pld_chirps - self.chirp_offset[0], self.chirp_offset[1] - self.chirp_offset[0] -12, 0), min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps)])
            print('B and C            ', [max(self.pld_chirps - self.chirp_offset[0], self.chirp_offset[1] - self.chirp_offset[0], 0), self.pld_chirps])

            print('Symbol Recovery C: ')
            print('A , B and C        ', [min(max(self.pld_chirps-self.chirp_offset[1], 0), 0), max(self.pld_chirps-self.chirp_offset[1], 0)])
            print('B and C            ', [max(self.pld_chirps-self.chirp_offset[1], 0), max(self.pld_chirps - self.chirp_offset[1] + self.chirp_offset[0], 0)])
            print('Free C:            ', [max(self.pld_chirps-self.chirp_offset[1]+self.chirp_offset[0], 0), self.pld_chirps])

        # Free A
        for idx in range(0, max(self.chirp_offset[0] - 12, 0)):
            bgn_index_a = self.pld_begins[0] + idx * self.d_samples_per_symbol
            end_index_a = bgn_index_a + self.d_samples_per_symbol

            chirp_a = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_a[0:len(self.packet_o[bgn_index_a:end_index_a])] = self.packet_o[bgn_index_a:end_index_a]
            chirp_index_a, chirp_max_a, chirp_bin_a = self.get_fft_bins(chirp_a, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

            self.pld_indexes[0].append(chirp_index_a)
            self.pld_bins[0].append(chirp_max_a)

        # A and B preamble
        for idx in range(max(self.chirp_offset[0] - 12, 0), max(min(self.chirp_offset[1]-12, self.chirp_offset[0]), 0)):
            bgn_index_a = self.pld_begins[0] + idx * self.d_samples_per_symbol
            end_index_a = bgn_index_a + self.d_samples_per_symbol

            chirp_a = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_a[0:len(self.packet_o[bgn_index_a:end_index_a])] = self.packet_o[bgn_index_a:end_index_a]
            chirp_index_a, chirp_max_a, chirp_bin_a = self.get_fft_bins(chirp_a, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)

            self.pld_indexes[0].append(chirp_index_a)
            self.pld_bins[0].append(chirp_max_a)

        # A and B preamble and C preamble
        for idx in range(max(min(self.chirp_offset[1] - 12, self.chirp_offset[0]), 0), min(self.chirp_offset[0], self.pld_chirps)):
            bgn_index_a = self.pld_begins[0] + idx * self.d_samples_per_symbol
            end_index_a = bgn_index_a + self.d_samples_per_symbol

            chirp_a = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_a[0:len(self.packet_o[bgn_index_a:end_index_a])] = self.packet_o[bgn_index_a:end_index_a]
            chirp_index_a, chirp_max_a, chirp_bin_a = self.get_fft_bins(chirp_a, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

            self.pld_indexes[0].append(chirp_index_a)
            self.pld_bins[0].append(chirp_max_a)

        # A and B payload
        for idx in range(min(self.chirp_offset[0], self.pld_chirps), min(max(self.chirp_offset[0], self.chirp_offset[1]- 12),self.pld_chirps)):
            # Save A, previous A, back A
            bgn_index_a = self.pld_begins[0] + idx * self.d_samples_per_symbol
            end_index_a = bgn_index_a + self.d_samples_per_symbol

            chirp_a = self.packet_o[bgn_index_a:end_index_a]
            chirp_a0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_a1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_a0[:self.samp_offset[0]] = chirp_a[:self.samp_offset[0]]
            chirp_a1[self.samp_offset[0]:] = chirp_a[self.samp_offset[0]:]

            chirp_index_a, chirp_max_a, chirp_bin_a = self.get_fft_bins(chirp_a, self.d_samples_per_symbol, self.detect_scope, self.sfd_peak-1)
            chirp_index_a0, chirp_max_a0, chirp_bin_a0 = self.get_fft_bins(chirp_a0, self.d_samples_per_symbol, self.detect_scope, self.sfd_peak-1)
            chirp_index_a1, chirp_max_a1, chirp_bin_a1 = self.get_fft_bins(chirp_a1, self.d_samples_per_symbol, self.detect_scope, self.sfd_peak-1)

            self.pld_indexes[0].append([chirp_index_a, chirp_index_a0, chirp_index_a1])
            self.pld_bins[0].append([chirp_max_a, chirp_max_a0, chirp_max_a1])

        # A and B payload and C preamble
        for idx in range(min(max(self.chirp_offset[0], self.chirp_offset[1] - 12), self.pld_chirps), min(self.chirp_offset[1], self.pld_chirps)):
            # Save A, previous A, back A
            bgn_index_a = self.pld_begins[0] + idx * self.d_samples_per_symbol
            end_index_a = bgn_index_a + self.d_samples_per_symbol

            chirp_a = self.packet_o[bgn_index_a:end_index_a]
            chirp_a0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_a1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_a0[:self.samp_offset[0]] = chirp_a[:self.samp_offset[0]]
            chirp_a1[self.samp_offset[0]:] = chirp_a[self.samp_offset[0]:]

            chirp_index_a, chirp_max_a, chirp_bin_a = self.get_fft_bins(chirp_a, self.d_samples_per_symbol, self.detect_scope, self.sfd_peak)
            chirp_index_a0, chirp_max_a0, chirp_bin_a0 = self.get_fft_bins(chirp_a0, self.d_samples_per_symbol, self.detect_scope, self.sfd_peak)
            chirp_index_a1, chirp_max_a1, chirp_bin_a1 = self.get_fft_bins(chirp_a1, self.d_samples_per_symbol, self.detect_scope, self.sfd_peak)

            self.pld_indexes[0].append([chirp_index_a, chirp_index_a0, chirp_index_a1])
            self.pld_bins[0].append([chirp_max_a, chirp_max_a0, chirp_max_a1])

        # A and B payload and C payload
        for idx in range(min(self.chirp_offset[1], self.pld_chirps), self.pld_chirps):
            # Save previous A, mid A, back A
            bgn_index_a = self.pld_begins[0] + idx * self.d_samples_per_symbol
            end_index_a = bgn_index_a + self.d_samples_per_symbol

            chirp_a = self.packet_o[bgn_index_a:end_index_a]
            chirp_a0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_a1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_a2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_a0[0:self.samp_offset[0]] = chirp_a[0:self.samp_offset[0]]
            chirp_a1[self.samp_offset[0]:self.samp_offset[1]] = chirp_a[self.samp_offset[0]:self.samp_offset[1]]
            chirp_a2[self.samp_offset[1]:] = chirp_a[self.samp_offset[1]:]

            chirp_index_a0, chirp_max_a0, chirp_bin_a0 = self.get_fft_bins(chirp_a0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
            chirp_index_a1, chirp_max_a1, chirp_bin_a1 = self.get_fft_bins(chirp_a1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
            chirp_index_a2, chirp_max_a2, chirp_bin_a2 = self.get_fft_bins(chirp_a2, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

            self.pld_indexes[0].append([chirp_index_a0, chirp_index_a1, chirp_index_a2])
            self.pld_bins[0].append([chirp_max_a0, chirp_max_a1, chirp_max_a2])

        # B and A payload
        for idx in range(0, max(self.chirp_offset[1] - self.chirp_offset[0] - 12, 0)):
            # Save B, previous B, back B
            bgn_index_b = self.pld_begins[1] + idx * self.d_samples_per_symbol
            end_index_b = bgn_index_b + self.d_samples_per_symbol

            chirp_b = self.packet_o[bgn_index_b:end_index_b]
            chirp_b0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_b1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_b0[:self.d_samples_per_symbol-self.samp_offset[0]] = chirp_b[:self.d_samples_per_symbol-self.samp_offset[0]]
            chirp_b1[self.d_samples_per_symbol-self.samp_offset[0]:] = chirp_b[self.d_samples_per_symbol-self.samp_offset[0]:]

            chirp_index_b, chirp_max_b, chirp_bin_b = self.get_fft_bins(chirp_b, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)
            chirp_index_b0, chirp_max_b0, chirp_bin_b0 = self.get_fft_bins(chirp_b0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)
            chirp_index_b1, chirp_max_b1, chirp_bin_b1 = self.get_fft_bins(chirp_b1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)

            self.pld_indexes[1].append([chirp_index_b, chirp_index_b0, chirp_index_b1])
            self.pld_bins[1].append([chirp_max_b, chirp_max_b0, chirp_max_b1])

        # B and A payload and C preamble
        for idx in range(max(self.chirp_offset[1]-self.chirp_offset[0]-12, 0), max(min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps-self.chirp_offset[0]), 0)):
            # Save B, previous B, back B
            bgn_index_b = self.pld_begins[1] + idx * self.d_samples_per_symbol
            end_index_b = bgn_index_b + self.d_samples_per_symbol

            chirp_b = self.packet_o[bgn_index_b:end_index_b]
            chirp_b0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_b1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_b0[:self.d_samples_per_symbol-self.samp_offset[0]] = chirp_b[:self.d_samples_per_symbol-self.samp_offset[0]]
            chirp_b1[self.d_samples_per_symbol-self.samp_offset[0]:] = chirp_b[self.d_samples_per_symbol-self.samp_offset[0]:]

            chirp_index_b, chirp_max_b, chirp_bin_b = self.get_fft_bins(chirp_b, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
            chirp_index_b0, chirp_max_b0, chirp_bin_b0 = self.get_fft_bins(chirp_b0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
            chirp_index_b1, chirp_max_b1, chirp_bin_b1 = self.get_fft_bins(chirp_b1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

            self.pld_indexes[1].append([chirp_index_b, chirp_index_b0, chirp_index_b1])
            self.pld_bins[1].append([chirp_max_b, chirp_max_b0, chirp_max_b1])

        # B and A payload and C payload
        for idx in range(max(min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps-self.chirp_offset[0]), 0), max(self.pld_chirps-self.chirp_offset[0], 0)):
            # Save previous B, mid B, back B
            bgn_index_b = self.pld_begins[1] + idx * self.d_samples_per_symbol
            end_index_b = bgn_index_b + self.d_samples_per_symbol

            chirp_b = self.packet_o[bgn_index_b:end_index_b]
            chirp_b0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_b1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_b2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_b0[0:self.samp_offset[1]-self.samp_offset[0]] = chirp_b[0:self.samp_offset[1]-self.samp_offset[0]]
            chirp_b1[self.samp_offset[1]-self.samp_offset[0]:self.d_samples_per_symbol-self.samp_offset[0]] = chirp_b[self.samp_offset[1]-self.samp_offset[0]:self.d_samples_per_symbol-self.samp_offset[0]]
            chirp_b2[self.d_samples_per_symbol-self.samp_offset[0]:] = chirp_b[self.d_samples_per_symbol-self.samp_offset[0]:]

            chirp_index_b0, chirp_max_b0, chirp_bin_b0 = self.get_fft_bins(chirp_b0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
            chirp_index_b1, chirp_max_b1, chirp_bin_b1 = self.get_fft_bins(chirp_b1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
            chirp_index_b2, chirp_max_b2, chirp_bin_b2 = self.get_fft_bins(chirp_b2, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

            self.pld_indexes[1].append([chirp_index_b0, chirp_index_b1, chirp_index_b2])
            self.pld_bins[1].append([chirp_max_b0, chirp_max_b1, chirp_max_b2])

        # B and C preamble
        for idx in range(max(self.pld_chirps - self.chirp_offset[0], self.chirp_offset[1] - self.chirp_offset[0] -12, 0), min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps)):
            bgn_index_b = self.pld_begins[1] + idx * self.d_samples_per_symbol
            end_index_b = bgn_index_b + self.d_samples_per_symbol

            chirp_b = self.packet_o[bgn_index_b:end_index_b]
            chirp_index_b, chirp_max_b, chirp_bin_b = self.get_fft_bins(chirp_b, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)

            self.pld_indexes[1].append(chirp_index_b)
            self.pld_bins[1].append(chirp_max_b)

        # B and C payload
        for idx in range(max(self.pld_chirps - self.chirp_offset[0], self.chirp_offset[1] - self.chirp_offset[0], 0), self.pld_chirps):
            # Save B, previous B, back B
            bgn_index_b = self.pld_begins[1] + idx * self.d_samples_per_symbol
            end_index_b = bgn_index_b + self.d_samples_per_symbol

            chirp_b = self.packet_o[bgn_index_b:end_index_b]
            chirp_b0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_b1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_b0[:self.samp_offset[1]-self.samp_offset[0]] = chirp_b[:self.samp_offset[1]-self.samp_offset[0]]
            chirp_b1[self.samp_offset[1]-self.samp_offset[0]:] = chirp_b[self.samp_offset[1]-self.samp_offset[0]:]

            chirp_index_b, chirp_max_b, chirp_bin_b = self.get_fft_bins(chirp_b, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)
            chirp_index_b0, chirp_max_b0, chirp_bin_b0 = self.get_fft_bins(chirp_b0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)
            chirp_index_b1, chirp_max_b1, chirp_bin_b1 = self.get_fft_bins(chirp_b1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)


            self.pld_indexes[1].append([chirp_index_b, chirp_index_b0, chirp_index_b1])
            self.pld_bins[1].append([chirp_max_b, chirp_max_b0, chirp_max_b1])

        # C and A payload and B payload
        for idx in range(min(max(self.pld_chirps-self.chirp_offset[1], 0), 0), max(self.pld_chirps-self.chirp_offset[1], 0)):
            # Save previous C, mid C, back C
            bgn_index_c = self.pld_begins[2] + idx * self.d_samples_per_symbol
            end_index_c = bgn_index_c + self.d_samples_per_symbol

            chirp_c = self.packet_o[bgn_index_c:end_index_c]
            chirp_c0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_c1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_c2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_c0[0:self.d_samples_per_symbol-self.samp_offset[1]] = chirp_c[0:self.d_samples_per_symbol-self.samp_offset[1]]
            chirp_c1[self.d_samples_per_symbol-self.samp_offset[1]:self.d_samples_per_symbol-self.samp_offset[1]+self.samp_offset[0]] = chirp_c[self.d_samples_per_symbol-self.samp_offset[1]:self.d_samples_per_symbol-self.samp_offset[1]+self.samp_offset[0]]
            chirp_c2[self.d_samples_per_symbol-self.samp_offset[1]+self.samp_offset[0]:] = chirp_c[self.d_samples_per_symbol-self.samp_offset[1]+self.samp_offset[0]:]

            chirp_index_c0, chirp_max_c0, chirp_bin_c0 = self.get_fft_bins(chirp_c0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
            chirp_index_c1, chirp_max_c1, chirp_bin_c1 = self.get_fft_bins(chirp_c1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)
            chirp_index_c2, chirp_max_c2, chirp_bin_c2 = self.get_fft_bins(chirp_c2, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

            self.pld_indexes[2].append([chirp_index_c0, chirp_index_c1, chirp_index_c2])
            self.pld_bins[2].append([chirp_max_c0, chirp_max_c1, chirp_max_c2])

        # C and B payload
        for idx in range(max(self.pld_chirps-self.chirp_offset[1], 0), max(self.pld_chirps - self.chirp_offset[1] + self.chirp_offset[0], 0)):
            # Save previous C, mid C, back C
            bgn_index_c = self.pld_begins[2] + idx * self.d_samples_per_symbol
            end_index_c = bgn_index_c + self.d_samples_per_symbol

            chirp_c = self.packet_o[bgn_index_c:end_index_c]
            chirp_c0 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_c1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)

            chirp_c0[:self.d_samples_per_symbol-self.samp_offset[1]+self.samp_offset[0]] = chirp_c[:self.d_samples_per_symbol-self.samp_offset[1]+self.samp_offset[0]]
            chirp_c1[self.d_samples_per_symbol-self.samp_offset[1]+self.samp_offset[0]:] = chirp_c[self.d_samples_per_symbol-self.samp_offset[1]+self.samp_offset[0]:]

            chirp_index_c, chirp_max_c, chirp_bin_c = self.get_fft_bins(chirp_c, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)
            chirp_index_c0, chirp_max_c0, chirp_bin_c0 = self.get_fft_bins(chirp_c0, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)
            chirp_index_c1, chirp_max_c1, chirp_bin_c1 = self.get_fft_bins(chirp_c1, self.d_samples_per_symbol, self.detect_scope, self.pld_peak-1)

            self.pld_indexes[2].append([chirp_index_c, chirp_index_c0, chirp_index_c1])
            self.pld_bins[2].append([chirp_max_c, chirp_max_c0, chirp_max_c1])

        # Free C
        for idx in range(max(self.pld_chirps-self.chirp_offset[1]+self.chirp_offset[0], 0), self.pld_chirps):
            bgn_index_c = self.pld_begins[2] + idx * self.d_samples_per_symbol
            end_index_c = bgn_index_c + self.d_samples_per_symbol

            chirp_c = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp_c[0:len(self.packet_o[bgn_index_c:end_index_c])] = self.packet_o[bgn_index_c:end_index_c]
            chirp_index_c, chirp_max_c, chirp_bin_c = self.get_fft_bins(chirp_c, self.d_samples_per_symbol, self.detect_scope, self.pld_peak)

            self.pld_indexes[2].append(chirp_index_c)
            self.pld_bins[2].append(chirp_max_c)


    def packet_decode(self):
        if len(self.sfd_begins) < 3:
            return

        # Symbol Recovery
        # Eliminate known preamble
        preamble_index_ab = (self.preamble_indexes[1] - self.preamble_indexes[0] +
                             self.d_quad_samples_per_symbol // self.d_decim_factor + self.d_number_of_bins) % self.d_number_of_bins
        preamble_index_ac = (self.preamble_indexes[2] - self.preamble_indexes[0] +
                             self.d_quad_samples_per_symbol // self.d_decim_factor + self.d_number_of_bins) % self.d_number_of_bins
        preamble_index_bc = (self.preamble_indexes[2] - self.preamble_indexes[1] +
                             self.d_quad_samples_per_symbol // self.d_decim_factor + self.d_number_of_bins) % self.d_number_of_bins

        if self.d_debug>=2:
            print('Known preamble : ', [preamble_index_ab, preamble_index_ac, preamble_index_bc])

        # free A
        for idx in range(0, max(self.chirp_offset[0]-12, 0)):
            self.rec_indexes[0].append(self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])])
            self.rec_bins[0].append(np.max(self.pld_bins[0][idx]))

        # A payload 与 B preamble 叠加，需要排除 B preamble index
        for idx in range(max(self.chirp_offset[0] - 12, 0), max(min(self.chirp_offset[1]-12, self.chirp_offset[0]), 0)):
            curr_index_0 = []
            curr_max_0 = []
            for jdx in range(len(self.pld_indexes[0][idx])):
                # 排除与preamble B相同的可能, preamble_index_ab
                if abs(self.pld_indexes[0][idx][jdx] - preamble_index_ab) > self.pld_tolerance \
                        and self.pld_bins[0][idx][jdx] > self.bin_threshold:
                    curr_index_0.append(self.pld_indexes[0][idx][jdx])
                    curr_max_0.append(self.pld_bins[0][idx][jdx])
            if len(curr_index_0) >= 1:
                self.rec_indexes[0].append(curr_index_0[np.argmax(curr_max_0)])
                self.rec_bins[0].append(np.max(curr_max_0))
            else:
                # 若检测不到，可能是与preamble重叠了，剩下payload里面最大的肯定是preamble index
                self.rec_indexes[0].append(self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])])
                self.rec_bins[0].append(np.max(self.pld_bins[0][idx]))

        # A payload 与 B C preamble叠加，需要排除 B、C preamble index
        for idx in range(max(min(self.chirp_offset[1] - 12, self.chirp_offset[0]), 0), min(self.chirp_offset[0], self.pld_chirps)):
            curr_index_0 = []
            curr_max_0 = []
            for jdx in range(len(self.pld_indexes[0][idx])):
                # 排除与preamble相同的可能, preamble_index_ab和preamble_index_ac
                if abs(self.pld_indexes[0][idx][jdx] - preamble_index_ab) > self.pld_tolerance \
                        and self.pld_bins[0][idx][jdx] > self.bin_threshold \
                        and abs(self.pld_indexes[0][idx][jdx] - preamble_index_ac) > self.pld_tolerance:
                    curr_index_0.append(self.pld_indexes[0][idx][jdx])
                    curr_max_0.append(self.pld_bins[0][idx][jdx])
            if len(curr_index_0) >= 1:
                self.rec_indexes[0].append(curr_index_0[np.argmax(curr_max_0)])
                self.rec_bins[0].append(np.max(curr_max_0))
            else:
                # 若检测不到，可能是与preamble B 或者 C 重叠了,
                # 取能量最大的Index
                self.rec_indexes[0].append(self.pld_indexes[0][idx][np.argmax(self.pld_bins[0][idx])])
                self.rec_bins[0].append(np.max(self.pld_bins[0][idx]))

        # A与B的payload叠加
        # 按分段进行Symbol Recovery
        # A中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps]
        # 1: [0, soff1]
        # 2: [soff1, dsps]
        samp_offset_dict = {1: self.samp_offset[0], 2: self.d_samples_per_symbol - self.samp_offset[0]}
        samp_offset_dict = np.array(sorted(samp_offset_dict.items(), key=lambda x: x[1]))
        ref_seg_idx = samp_offset_dict[-1][0]

        # 只有A和B的payload,C包还未到达
        for idx in range(min(self.chirp_offset[0], self.pld_chirps), min(max(self.chirp_offset[0], self.chirp_offset[1]- 12),self.pld_chirps)):
            # 挑出3次重复的index
            curr_idx = self.pld_indexes[0][idx][ref_seg_idx]
            curr_max = self.pld_bins[0][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[0][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[0][idx][jdx][kdx]) <= self.pld_tolerance and self.pld_bins[0][idx][jdx][kdx] > 0:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[0][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3
            if len(rpt_index) == 1: # 理想情况
                self.rec_indexes[0].append(curr_idx[rpt_index[0]])
                self.rec_bins[0].append(curr_max[rpt_index[0]])
            else:
                # A.B都满足cnt>=3
                if len(rpt_index) > 1:
                    self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                # A.B都不满足cnt>=3，则加入cnt>=2，后续cross decoding或power mapping
                else:
                    self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # A的payload 与 B的payload 与 C的preamble 叠加
        # 在剔除preamble_index_ac的过程中统计出现3次的index
        for idx in range(min(max(self.chirp_offset[0], self.chirp_offset[1] - 12), self.pld_chirps), min(self.chirp_offset[1], self.pld_chirps)):
            # 挑出3次重复的index
            curr_idx = self.pld_indexes[0][idx][ref_seg_idx]
            curr_max = self.pld_bins[0][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[0][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        # 剔除preamble_index_ac
                        if abs(curr_idx[ldx] - self.pld_indexes[0][idx][jdx][kdx]) <= self.pld_tolerance < abs(curr_idx[ldx] - preamble_index_ac) \
                                and self.pld_bins[0][idx][jdx][kdx] > 0:
                            curr_cnt[ldx] += 1
                            break

            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3
            if len(rpt_index) == 1: # 理想情况
                self.rec_indexes[0].append(curr_idx[rpt_index[0]])
                self.rec_bins[0].append(curr_max[rpt_index[0]])
            else:
                # A.B都满足cnt>=3
                if len(rpt_index) > 1:
                    self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                # A.B都不满足cnt>=3，则加入cnt>=2，后续cross decoding或power mapping
                else:
                    self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # 解A所有其它的 A payload + B payload + C payload
        # A中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, soff1]
        # 1: [soff1, soff2]
        # 2: [soff2, dsps]
        samp_offset_dict = {0: self.samp_offset[0], 1: self.samp_offset[1] - self.samp_offset[0], 2: self.d_samples_per_symbol - self.samp_offset[1]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]

        for idx in range(min(self.chirp_offset[1], self.pld_chirps), self.pld_chirps):
            # 挑出3次重复的index
            curr_idx = self.pld_indexes[0][idx][ref_seg_idx]
            curr_max = self.pld_bins[0][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[0][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[0][idx][jdx][kdx]) <= self.pld_tolerance and self.pld_bins[0][idx][jdx][kdx] > 0:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[0][idx][jdx][kdx])
            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3 three

            if len(rpt_index) == 1:
                self.rec_indexes[0].append(curr_idx[rpt_index[0]])
                self.rec_bins[0].append(curr_max[rpt_index[0]])
            else:
                # 多个都满足cnt>=3
                if len(rpt_index) > 1:
                    self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                # 都不满足cnt>=3，则加入cnt>=2，后续cross decoding或power mapping
                else:
                    self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # 开始解B
        # B与A的payload叠加
        # B中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps]
        # 1: [0, dsps-soff0]
        # 2: [dsps-soff0]
        samp_offset_dict = {1: self.d_samples_per_symbol - self.samp_offset[0], 2: self.samp_offset[0]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]

        # 只有B和A的Payload
        for idx in range(0, max(self.chirp_offset[1] - self.chirp_offset[0] - 12, 0)):
            #求重复三次的index
            curr_idx = self.pld_indexes[1][idx][ref_seg_idx]
            curr_max = self.pld_bins[1][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance and self.pld_bins[1][idx][jdx][kdx] > 0:
                            curr_cnt[ldx] += 1
                            break
            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3 three
            if len(rpt_index) == 1:
                self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                self.rec_bins[1].append(curr_max[rpt_index[0]])
            else:
                if len(rpt_index)>1:
                    self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                else:
                    self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # B与 A payload、C preamble部分
        for idx in range(max(self.chirp_offset[1]-self.chirp_offset[0]-12, 0), max(min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps-self.chirp_offset[0]), 0)):
            #求重复三次的index
            curr_idx = self.pld_indexes[1][idx][ref_seg_idx]
            curr_max = self.pld_bins[1][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        # 剔除C的preamble
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance < abs(curr_idx[ldx] - preamble_index_bc) \
                                and self.pld_bins[1][idx][jdx][kdx] > 0:
                            curr_cnt[ldx] += 1
                            break
            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3 three
            if len(rpt_index) == 1:
                self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                self.rec_bins[1].append(curr_max[rpt_index[0]])
            else:
                if len(rpt_index)>1:
                    self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                else:
                    self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # B 的payload叠加部分 ABC
        # B中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, soff2-soff1]
        # 1: [soff2-soff1, dsps-soff1]
        # 2: [dsps-soff1, dsps]
        samp_offset_dict = {0: self.samp_offset[1] - self.samp_offset[0], 1: self.d_samples_per_symbol - self.samp_offset[1], 2: self.samp_offset[0]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]
        for idx in range(max(min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps-self.chirp_offset[0]), 0), max(self.pld_chirps-self.chirp_offset[0], 0)):
            #求重复三次的index
            curr_idx = self.pld_indexes[1][idx][ref_seg_idx]
            curr_max = self.pld_bins[1][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance \
                                and self.pld_bins[1][idx][jdx][kdx]>0:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[1][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]
            if len(rpt_index) == 1:
                self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                self.rec_bins[1].append(curr_max[rpt_index[0]])
            else:
                if len(rpt_index)>1:
                    self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                else:
                    self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # B 的payload叠加部分
        # B中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps]
        # 1: [0, soff2-soff1]
        # 2: [soff2-soff1, dsps]
        samp_offset_dict = {1: self.samp_offset[1] - self.samp_offset[0], 2: self.d_samples_per_symbol - self.samp_offset[1] + self.samp_offset[0]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]

        # B and C preamble
        for idx in range(max(self.pld_chirps - self.chirp_offset[0], self.chirp_offset[1] - self.chirp_offset[0] -12, 0), min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps)):
            curr_idx = []
            curr_max = []
            for jdx in range(len(self.pld_indexes[1][idx])):
                if abs(self.pld_indexes[1][idx][jdx] - preamble_index_bc) > self.pld_tolerance \
                        and self.pld_bins[1][idx][jdx] > self.bin_threshold:
                    curr_idx.append(self.pld_indexes[1][idx][jdx])
                    curr_max.append(self.pld_bins[1][idx][jdx])

            if len(curr_idx) >= 1:
                self.rec_indexes[1].append(curr_idx[np.argmax(curr_max)])
                self.rec_bins[1].append(np.max(curr_max))
            else:
                # 若检测不到，可能是与preamble重叠了，剩下payload里面最大的肯定是preamble index
                self.rec_indexes[1].append(self.pld_indexes[1][idx][np.argmax(self.pld_bins[1][idx])])
                self.rec_bins[1].append(np.max(self.pld_bins[1][idx]))

        # B and C payload
        for idx in range(max(self.pld_chirps - self.chirp_offset[0], self.chirp_offset[1] - self.chirp_offset[0], 0), self.pld_chirps):
            #求重复三次的index
            curr_idx = self.pld_indexes[1][idx][ref_seg_idx]
            curr_max = self.pld_bins[1][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance \
                                and self.pld_bins[1][idx][jdx][kdx]>0:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[1][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]
            if len(rpt_index) == 1:
                self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                self.rec_bins[1].append(curr_max[rpt_index[0]])
            else:
                if len(rpt_index)>1:
                    self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                else:
                    self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # C 的payload叠加部分, ABC
        # C中存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps-soff2]
        # 1: [dsps-soff2, dsps-soff2+soff1]
        # 2: [dsps-soff2+soff1, dsps]
        samp_offset_dict = {0: self.d_samples_per_symbol - self.samp_offset[1], 1: self.samp_offset[0], 2: self.samp_offset[1] - self.samp_offset[0]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]
        for idx in range(min(max(self.pld_chirps-self.chirp_offset[1], 0), 0), max(self.pld_chirps-self.chirp_offset[1], 0)):
            # 求重复三次的index
            curr_idx = self.pld_indexes[2][idx][ref_seg_idx]
            curr_max = self.pld_bins[2][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[2][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[2][idx][jdx][kdx]) <= self.pld_tolerance \
                                and self.pld_bins[2][idx][jdx][kdx]>0:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[1][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]
            if len(rpt_index) == 1:
                self.rec_indexes[2].append(curr_idx[rpt_index[0]])
                self.rec_bins[2].append(curr_max[rpt_index[0]])
            else:
                if len(rpt_index)>1:
                    self.rec_indexes[2].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[2].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                else:
                    # 这里可能存了多个，后面需要用cross decoding和power mapping解决；
                    self.rec_indexes[2].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[2].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # C的payload部分，BC
        # C中存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps]
        # 1: [0, dsps-soff2+soff1]
        # 2: [dsps-soff2+soff1, dsps]
        samp_offset_dict = {1: self.d_samples_per_symbol - self.samp_offset[1] + self.samp_offset[0], 2: self.samp_offset[1] - self.samp_offset[0]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]
        for idx in range(max(self.pld_chirps-self.chirp_offset[1], 0), max(self.pld_chirps - self.chirp_offset[1] + self.chirp_offset[0], 0)):
            # 求重复三次的index
            curr_idx = self.pld_indexes[2][idx][ref_seg_idx]
            curr_max = self.pld_bins[2][idx][ref_seg_idx]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[2][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[2][idx][jdx][kdx]) <= self.pld_tolerance \
                                and self.pld_bins[2][idx][jdx][kdx] > 0:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[1][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]
            if len(rpt_index) == 1:
                self.rec_indexes[2].append(curr_idx[rpt_index[0]])
                self.rec_bins[2].append(curr_max[rpt_index[0]])
            else:
                if len(rpt_index)>1:
                    self.rec_indexes[2].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                    self.rec_bins[2].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])
                else:
                    # 这里可能存了多个，后面需要用cross decoding和power mapping解决；
                    self.rec_indexes[2].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                    self.rec_bins[2].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # free C
        for idx in range(max(self.pld_chirps-self.chirp_offset[1]+self.chirp_offset[0], 0), self.pld_chirps):
            self.rec_indexes[2].append(self.pld_indexes[2][idx][np.argmax(self.pld_bins[2][idx])])
            self.rec_bins[2].append(np.max(self.pld_bins[2][idx]))

        for idx in range(len(self.rec_indexes)):
            self.rec_results[idx] = self.rec_indexes[idx].copy()

        # Cross decoding
        # 初始化参数
        # bin_offset of B/C when aligning with A/B
        bin_offset_ab = self.samp_offset[0] // self.d_decim_factor
        bin_offset_ac = self.samp_offset[1] // self.d_decim_factor
        bin_offset_bc = (self.samp_offset[1] - self.samp_offset[0]) // self.d_decim_factor

        # sfd_bin Order for power mapping
        sfd_bins_dict = {0: self.sfd_bins[0], 1: self.sfd_bins[1], 2: self.sfd_bins[2]}
        sfd_bins_dict = np.array(sorted(sfd_bins_dict.items(), key=lambda x:x[1], reverse=True))
        sfd_bin_order = [0, 1, 2]
        for idx in range(len(self.sfd_bins)):
            sfd_bin_order[int(sfd_bins_dict[idx][0])] = idx

        # A与B的payload叠加
        # 按分段进行Cross decoding和Power mapping
        # A中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps]
        # 1: [0, soff1]
        # 2: [soff1, dsps]
        samp_offset_dict = {1: self.samp_offset[0], 2: self.d_samples_per_symbol - self.samp_offset[0]}
        samp_offset_dict = np.array(sorted(samp_offset_dict.items(), key=lambda x: x[1]))
        ref_seg_idx = samp_offset_dict[-1][0]

        # 只有A和B的payload,C包还未到达
        for idx in range(min(self.chirp_offset[0], self.pld_chirps), min(max(self.chirp_offset[0], self.chirp_offset[1]- 12),self.pld_chirps)):
            if not (type(self.rec_indexes[0][idx]) is int):
                curr_idx = []
                oly_idxes = []
                oly_pld_b = idx - self.chirp_offset[0] - 2 + ref_seg_idx
                if 0 <= oly_pld_b < max(0, self.chirp_offset[1] - self.chirp_offset[0] - 12):
                    if type(self.rec_indexes[1][oly_pld_b]) is int:
                        oly_idxes.append((self.rec_indexes[1][oly_pld_b] - bin_offset_ab + self.d_number_of_bins) % self.d_number_of_bins)

                for jdx in range(len(self.rec_indexes[0][idx])):
                    is_oly = False
                    for oly_idx in oly_idxes:
                        if abs(self.rec_indexes[0][idx][jdx] - oly_idx) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[0][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[0][idx] = curr_idx[0]
                else:
                    if self.sfd_bins[0] > self.sfd_bins[1]:
                        jdx = 0
                    else:
                        jdx = 1

                    self.rec_indexes[0][idx] = self.pld_indexes[0][idx][ref_seg_idx][jdx]
                    self.rec_bins[0][idx] = self.pld_bins[0][idx][ref_seg_idx][jdx]

        # A的payload 与 B的payload 与 C的preamble 叠加
        for idx in range(min(max(self.chirp_offset[0], self.chirp_offset[1] - 12), self.pld_chirps), min(self.chirp_offset[1], self.pld_chirps)):
            if not (type(self.rec_indexes[0][idx]) is int):
                # 剩余小于一个, 说明A = Pre C
                if len(self.rec_indexes[0][idx]) <= 1:
                    found = False
                    for jdx in range(len(self.pld_indexes[0][idx][0])):
                        if abs(self.pld_indexes[0][idx][0][jdx] - preamble_index_ac) <= self.pld_tolerance:
                            self.rec_indexes[0][idx] = self.pld_indexes[0][idx][0][jdx]
                            found = True
                            break
                    if not found:
                        self.rec_indexes[0][idx] = preamble_index_ac
                    continue
                # 剩余两个以上
                else:
                    curr_idx = []
                    oly_idxes = []

                    oly_pld_b = idx - self.chirp_offset[0] - 2 + ref_seg_idx
                    if max(0, self.chirp_offset[1] - self.chirp_offset[0] - 12) <= oly_pld_b < self.chirp_offset[1] - self.chirp_offset[0]:
                        if type(self.rec_indexes[1][oly_pld_b]) is int:
                            oly_idxes.append((self.rec_indexes[1][oly_pld_b] - bin_offset_ab + self.d_number_of_bins) % self.d_number_of_bins)

                    for jdx in range(len(self.rec_indexes[0][idx])):
                        is_oly = False
                        for oly_idx in oly_idxes:
                            if abs(self.rec_indexes[0][idx][jdx] - oly_idx) < self.pld_tolerance:
                                is_oly = True
                                break
                        if not is_oly:
                            curr_idx.append(self.rec_indexes[0][idx][jdx])

                    if len(curr_idx) == 1:
                        self.rec_indexes[0][idx] = curr_idx[0]
                    else:
                        self.rec_indexes[0][idx] = self.pld_indexes[0][idx][ref_seg_idx][sfd_bin_order[0]]
                        self.rec_bins[0][idx] = self.pld_bins[0][idx][ref_seg_idx][sfd_bin_order[0]]

        # 解A所有其它的 A payload + B payload + C payload
        # A中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, soff1]
        # 1: [soff1, soff2]
        # 2: [soff2, dsps]
        samp_offset_dict = {0: self.samp_offset[0], 1: self.samp_offset[1] - self.samp_offset[0], 2: self.d_samples_per_symbol - self.samp_offset[1]}
        if self.d_debug >= 2:
            print('Segments A: ', samp_offset_dict)
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]

        for idx in range(min(self.chirp_offset[1], self.pld_chirps), self.pld_chirps):
            if not (type(self.rec_indexes[0][idx]) is int):
                curr_idx = []
                oly_idxes = []

                oly_pld_b = idx - self.chirp_offset[0] - 2 +  1 if ref_seg_idx<1 else 2
                oly_pld_c = idx - self.chirp_offset[1] - 2 +  1 if ref_seg_idx<2 else 2

                if (self.chirp_offset[1] - self.chirp_offset[0]) <= oly_pld_b < self.pld_chirps - self.chirp_offset[0]:
                    if type(self.rec_indexes[1][oly_pld_b]) is int:
                        oly_idxes.append((self.rec_indexes[1][oly_pld_b] - bin_offset_ab + self.d_number_of_bins) % self.d_number_of_bins)

                if 0 <= oly_pld_c < self.pld_chirps - self.chirp_offset[1]:
                    if type(self.rec_indexes[2][oly_pld_c]) is int:
                        oly_idxes.append((self.rec_indexes[2][oly_pld_c] - bin_offset_ac + self.d_number_of_bins) % self.d_number_of_bins)

                for jdx in range(len(self.rec_indexes[0][idx])):
                    is_oly = False
                    for oly_idx in oly_idxes:
                        if abs(self.rec_indexes[0][idx][jdx] - oly_idx) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[0][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[0][idx] = curr_idx[0]
                else:
                    self.rec_indexes[0][idx] = self.pld_indexes[0][idx][ref_seg_idx][sfd_bin_order[0]]
                    self.rec_bins[0][idx] = self.pld_bins[0][idx][ref_seg_idx][sfd_bin_order[0]]

        # 开始解B
        # B与A的payload叠加
        # B中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps]
        # 1: [0, dsps-soff0]
        # 2: [dsps-soff0]
        samp_offset_dict = {1: self.d_samples_per_symbol - self.samp_offset[0], 2: self.samp_offset[0]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]

        # 只有B和A的Payload
        for idx in range(0, max(self.chirp_offset[1] - self.chirp_offset[0] - 12, 0)):
            if not (type(self.rec_indexes[1][idx]) is int):
                curr_idx = []
                oly_idxes = []

                oly_pld_a = idx + self.chirp_offset[0] - 1 + ref_seg_idx
                # 这里可以等于边界值
                if self.chirp_offset[0] <= oly_pld_a <= max(self.chirp_offset[1]-12, 0):
                    if type(self.rec_indexes[0][oly_pld_a]) is int :
                        oly_idxes.append((self.rec_indexes[0][oly_pld_a] + bin_offset_ab + self.d_number_of_bins) % self.d_number_of_bins)

                for jdx in range(len(self.rec_indexes[1][idx])):
                    is_oly = False  # overlay
                    for kdx in range(len(oly_idxes)):
                        if abs(self.rec_indexes[1][idx][jdx] - oly_idxes[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[1][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[1][idx] = curr_idx[0]
                else:
                    if self.sfd_bins[0] > self.sfd_bins[1]:
                        jdx=1
                    else:
                        jdx=0

                    self.rec_indexes[1][idx] = self.pld_indexes[1][idx][ref_seg_idx][jdx]
                    self.rec_bins[1][idx] = self.pld_bins[1][idx][ref_seg_idx][jdx]

        # B与 A payload、C preamble部分
        for idx in range(max(self.chirp_offset[1]-self.chirp_offset[0]-12, 0), max(min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps-self.chirp_offset[0]), 0)):
            if not (type(self.rec_indexes[1][idx]) is int):
                # 剩余小于一个, 说明B = Pre C
                if len(self.rec_indexes[1][idx]) <= 1:
                    found = False
                    for jdx in range(len(self.pld_indexes[1][idx][0])):
                        if abs(self.pld_indexes[1][idx][0][jdx] - preamble_index_bc) <= self.pld_tolerance:
                            self.rec_indexes[1][idx] = self.pld_indexes[1][idx][0][jdx]
                            found = True
                            break
                    if not found:
                        self.rec_indexes[1][idx] = preamble_index_bc
                    continue
                # 剩余两个以上
                else:
                    curr_idx = []
                    oly_idxes = []

                    oly_pld_a = idx + self.chirp_offset[0] - 1 + ref_seg_idx
                    # 这里可以等于边界值
                    if self.chirp_offset[0] <= oly_pld_a <= max(self.chirp_offset[1] - 12, 0):
                        if type(self.rec_indexes[0][oly_pld_a]) is int:
                            oly_idxes.append((self.rec_indexes[0][oly_pld_a] + bin_offset_ab + self.d_number_of_bins) % self.d_number_of_bins)

                    for jdx in range(len(self.rec_indexes[1][idx])):
                        is_oly = False  # overlay
                        for kdx in range(len(oly_idxes)):
                            if abs(self.rec_indexes[1][idx][jdx] - oly_idxes[kdx]) < self.pld_tolerance:
                                is_oly = True
                                break
                        if not is_oly:
                            curr_idx.append(self.rec_indexes[1][idx][jdx])

                    if len(curr_idx) == 1:
                        self.rec_indexes[1][idx] = curr_idx[0]
                    else:
                        self.rec_indexes[1][idx] = self.pld_indexes[1][idx][ref_seg_idx][sfd_bin_order[1]]
                        self.rec_bins[1][idx] = self.pld_bins[1][idx][ref_seg_idx][sfd_bin_order[1]]

        # B 的payload叠加部分 ABC
        # B中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, soff2-soff1]
        # 1: [soff2-soff1, dsps-soff1]
        # 2: [dsps-soff1, dsps]
        samp_offset_dict = {0: self.samp_offset[1] - self.samp_offset[0], 1: self.d_samples_per_symbol - self.samp_offset[1], 2: self.samp_offset[0]}
        if self.d_debug >= 2:
            print('Segments B: ', samp_offset_dict)
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]
        for idx in range(max(min(self.chirp_offset[1] - self.chirp_offset[0], self.pld_chirps-self.chirp_offset[0]), 0), max(self.pld_chirps-self.chirp_offset[0], 0)):
            if not (type(self.rec_indexes[1][idx]) is int):
                curr_idx = []
                oly_idxes = []

                oly_pld_a = idx + self.chirp_offset[0] - 1 + 1 if ref_seg_idx<2 else 2
                oly_pld_c = idx - self.chirp_offset[1] + self.chirp_offset[0] -2 + 1 if ref_seg_idx<1 else 2

                if self.chirp_offset[1] <= oly_pld_a < self.pld_chirps:
                    if type(self.rec_indexes[0][oly_pld_a]) is int:
                        oly_idxes.append((self.rec_indexes[0][oly_pld_a] + bin_offset_ab + self.d_number_of_bins) % self.d_number_of_bins)

                if 0 <= oly_pld_c < self.pld_chirps-self.chirp_offset[1]:
                    if type(self.rec_indexes[2][oly_pld_c]) is int:
                        oly_idxes.append((self.rec_indexes[2][oly_pld_c] - bin_offset_bc + self.d_number_of_bins) % self.d_number_of_bins)

                for jdx in range(len(self.rec_indexes[1][idx])):
                    is_oly = False  # overlay
                    for kdx in range(len(oly_idxes)):
                        if abs(self.rec_indexes[1][idx][jdx] - oly_idxes[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[1][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[1][idx] = curr_idx[0]
                else:
                    self.rec_indexes[1][idx] = self.pld_indexes[1][idx][ref_seg_idx][sfd_bin_order[1]]
                    self.rec_bins[1][idx] = self.pld_bins[1][idx][ref_seg_idx][sfd_bin_order[1]]

        # B 的payload叠加部分, BC
        # B中 存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps]
        # 1: [0, soff2-soff1]
        # 2: [soff2-soff1, dsps]
        samp_offset_dict = {1: self.samp_offset[1] - self.samp_offset[0], 2: self.d_samples_per_symbol - self.samp_offset[1] + self.samp_offset[0]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]
        for idx in range(max(self.pld_chirps - self.chirp_offset[0], self.chirp_offset[1] - self.chirp_offset[0], 0), self.pld_chirps):
            if not (type(self.rec_indexes[1][idx]) is int):
                curr_idx = []
                oly_idxes = []

                oly_pld_c = idx - self.chirp_offset[1] + self.chirp_offset[0] - 2 + ref_seg_idx

                # 这里可以超出边界值
                if self.pld_chirps - self.chirp_offset[1] -1 <= oly_pld_c <= self.pld_chirps-self.chirp_offset[1] + self.chirp_offset[0]:
                    if type(self.rec_indexes[2][oly_pld_c]) is int:
                        oly_idxes.append((self.rec_indexes[2][oly_pld_c] - bin_offset_bc + self.d_number_of_bins) % self.d_number_of_bins)

                for jdx in range(len(self.rec_indexes[1][idx])):
                    is_oly = False  # overlay
                    for kdx in range(len(oly_idxes)):
                        if abs(self.rec_indexes[1][idx][jdx] - oly_idxes[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[1][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[1][idx] = curr_idx[0]
                else:
                    if self.sfd_bins[1] > self.sfd_bins[2]:
                        jdx = 0
                    else:
                        jdx = 1

                    self.rec_indexes[1][idx] = self.pld_indexes[1][idx][ref_seg_idx][jdx]
                    self.rec_bins[1][idx] = self.pld_bins[1][idx][ref_seg_idx][jdx]

        # C 的payload叠加部分, ABC
        # C中存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps-soff2]
        # 1: [dsps-soff2, dsps-soff2+soff1]
        # 2: [dsps-soff2+soff1, dsps]
        samp_offset_dict = {0: self.d_samples_per_symbol - self.samp_offset[1], 1: self.samp_offset[0], 2: self.samp_offset[1] - self.samp_offset[0]}
        if self.d_debug >= 2:
            print('Segments C: ', samp_offset_dict)
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]
        for idx in range(min(max(self.pld_chirps-self.chirp_offset[1], 0), 0), max(self.pld_chirps-self.chirp_offset[1], 0)):
            if not (type(self.rec_indexes[2][idx]) is int):
                curr_idx = []
                oly_idxes = []

                oly_pld_a = idx + self.chirp_offset[1] - 1 + 1 if ref_seg_idx < 1 else 2
                oly_pld_b = idx + self.chirp_offset[1] - self.chirp_offset[0] -1 + 1 if ref_seg_idx < 2 else 2

                if self.chirp_offset[1] <= oly_pld_a < self.pld_chirps:
                    if type(self.rec_indexes[0][oly_pld_a]) is int:
                        oly_idxes.append((self.rec_indexes[0][oly_pld_a] + bin_offset_ac + self.d_number_of_bins) % self.d_number_of_bins)

                # 可以等于边界值
                if self.chirp_offset[1] - self.chirp_offset[0] <= oly_pld_b <= self.pld_chirps - self.chirp_offset[0]:
                    if type(self.rec_indexes[1][oly_pld_b]) is int:
                        oly_idxes.append((self.rec_indexes[1][oly_pld_b] + bin_offset_bc + self.d_number_of_bins) % self.d_number_of_bins)

                for jdx in range(len(self.rec_indexes[2][idx])):
                    is_oly = False  # overlay
                    for kdx in range(len(oly_idxes)):
                        if abs(self.rec_indexes[2][idx][jdx] - oly_idxes[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[2][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[2][idx] = curr_idx[0]
                else:
                    self.rec_indexes[2][idx] = self.pld_indexes[2][idx][ref_seg_idx][sfd_bin_order[2]]
                    self.rec_bins[2][idx] = self.pld_bins[2][idx][ref_seg_idx][sfd_bin_order[2]]


        # C的payload部分，BC
        # C中存的3个segment的宽度
        # self.d_samples_per_symbol -> dsps
        # 0: [0, dsps]
        # 1: [0, dsps-soff2+soff1]
        # 2: [dsps-soff2+soff1, dsps]
        samp_offset_dict = {1: self.d_samples_per_symbol - self.samp_offset[1] + self.samp_offset[0], 2: self.samp_offset[1] - self.samp_offset[0]}
        samp_offset_dict = sorted(samp_offset_dict.items(), key=lambda x: x[1])
        ref_seg_idx = samp_offset_dict[-1][0]
        for idx in range(max(self.pld_chirps-self.chirp_offset[1], 0), max(self.pld_chirps - self.chirp_offset[1] + self.chirp_offset[0], 0)):
            if not (type(self.rec_indexes[2][idx]) is int):
                curr_idx = []
                oly_idxes = []

                oly_pld_b = idx + self.chirp_offset[1] - self.chirp_offset[0] - 1 + ref_seg_idx

                if self.pld_chirps-self.chirp_offset[0] <= oly_pld_b < self.pld_chirps:
                    if type(self.rec_indexes[1][oly_pld_b]) is int:
                        oly_idxes.append((self.rec_indexes[1][oly_pld_b] + bin_offset_bc + self.d_number_of_bins) % self.d_number_of_bins)

                for jdx in range(len(self.rec_indexes[2][idx])):
                    is_oly = False  # overlay
                    for kdx in range(len(oly_idxes)):
                        if abs(self.rec_indexes[2][idx][jdx] - oly_idxes[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[2][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[2][idx] = curr_idx[0]
                else:
                    if self.sfd_bins[1] > self.sfd_bins[2]:
                        jdx = 1
                    else:
                        jdx = 0

                    self.rec_indexes[2][idx] = self.pld_indexes[2][idx][ref_seg_idx][jdx]
                    self.rec_bins[2][idx] = self.pld_bins[2][idx][ref_seg_idx][jdx]

        self.error_correct()
        self.get_chirp_error()
        self.recover_byte()
        self.get_byte_error()
        self.save_result()
        self.show_result()

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

        # while len(curr_peaks) < 3:
        temp_cnt = -1
        # 字典键值还是不能相同
        while len(curr_peaks) < num:
            curr_peaks.update({temp_cnt: 0})
            temp_cnt -= 1
        sorted_peaks = sorted(curr_peaks.items(), key=lambda kv: kv[1])
        max_idx = [sorted_peaks[-idx][0] for idx in range(1, num + 1)]
        max_bin = [sorted_peaks[-idx][1] for idx in range(1, num + 1)]
        # max_peaks = [(sorted_peaks[-idx][0], sorted_peaks[-idx][1]) for idx in range(3)]

        return max_idx, max_bin

    def detect_preamble_all(self, idx_stacks, bin_stacks):
        # 全匹配
        pre_idxes = np.full(len(idx_stacks), -1)
        pre_bins = np.full(len(idx_stacks), -1, dtype=float)
        for idx in range(len(idx_stacks)):
            pre_idx = idx_stacks[idx].bottom()
            pre_found = True

            curr_idx = []
            curr_pos = []  # 这个似乎没用
            for jdx in range(self.preamble_chirps):
                curr_found = False
                for kdx in range(len(idx_stacks)):
                    # 频率在附近，强度超过threshold，就认为是一个新频率。
                    if abs(pre_idx - idx_stacks[kdx].get_i(jdx)) < self.preamble_tolerance and \
                            bin_stacks[kdx].get_i(jdx) > self.bin_threshold:
                        curr_idx.append((jdx, kdx))
                        curr_pos.append(kdx)
                        curr_found = True
                        break
                if not curr_found:
                    pre_found = False
                    break
                # 所以fail 了怎么处理呢。。。
                # if not curr_found or (len(np.unique(curr_pos)) >= 3):
                #     pre_found = False
                #     break

            if pre_found:
                pre_idxes[idx] = pre_idx
                pre_bins[idx] = np.average([bin_stacks[bin_idx[1]].get_i(bin_idx[0]) for bin_idx in curr_idx])

        # 每个preamble 的频率和index
        return pre_idxes, pre_bins

    # 这个似乎没有用过，好像有点。。问题？
    def detect_preamble(self, idx_stacks, bin_stacks):
        pre_idxes = np.full(len(idx_stacks), -1)
        pre_bins = np.full(len(idx_stacks), -1, dtype=float)
        for idx in range(len(idx_stacks)):
            pre_idx = idx_stacks[idx].bottom()
            pre_found = True

            # 拿bottom的和其它比误差。
            for jdx in range(self.preamble_chirps):
                if abs(pre_idx - idx_stacks[idx].get_i(jdx)) >= self.preamble_tolerance \
                        or bin_stacks[idx].get_i(jdx) <= self.bin_threshold:
                    pre_found = False
                    break

            if pre_found:
                pre_idxes[idx] = pre_idx
                pre_bins[idx] = np.average(bin_stacks[idx].get_list())

        return pre_idxes, pre_bins

    def detect_down_chirp(self, bgn_idx, scope, num):
        # 后面0.25个是没有检测的

        self.sfd_history.clear()
        self.sfdbin_history.clear()
        self.sfdbins_history.clear()

        for idx in range(self.sfd_chirps):
            pad_chirp = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            curr_chirp = self.packet_o[
                         bgn_idx + idx * self.d_samples_per_symbol:bgn_idx + (idx + 1) * self.d_samples_per_symbol]
            pad_chirp[0:len(curr_chirp)] = curr_chirp  # 补齐0
            sfd_idx, sfd_max, sfd_bin = self.get_down_chirp_bin(pad_chirp, scope, num)
            self.sfd_history.append(sfd_idx)
            self.sfdbin_history.append(sfd_max)
            self.sfdbins_history.append(sfd_bin)

        sfd_found = True

        # 不是全匹配，就是匹配
        curr_idx = []
        for idx in range(self.sfd_chirps):
            curr_found = False
            for jdx in range(self.sfd_peak):
                if abs(self.sfd_history[idx][jdx] - self.d_half_number_of_bins) <= self.sfd_tolerance \
                        and self.sfdbin_history[idx][jdx] > self.sfd_threshold:
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

    def get_sfd_bin(self, scope, num):
        self.sfd_begins = []
        self.sfd_bins = []

        sfd_bgn_index = 10 * self.d_samples_per_symbol
        bgn_indexes = [sfd_bgn_index, sfd_bgn_index+self.time_offset[0], sfd_bgn_index+self.time_offset[1]]
        self.preamble_indexes = [0, (self.d_number_of_bins - self.time_offset[0]//self.d_decim_factor) % self.d_number_of_bins, (self.d_number_of_bins - self.time_offset[1]//self.d_decim_factor) % self.d_number_of_bins]


        sfd_found_all = True
        for idx in range(len(bgn_indexes)):
            if not self.detect_down_chirp(bgn_indexes[idx], scope, num):
                sfd_found_all = False
                break
            else:
                self.sfd_begins.append(bgn_indexes[idx])
                self.preamble_bins.append(self.sfd_bins[idx])

        return sfd_found_all

    def error_correct(self):
        # error correct
        index_error = np.array([0, 0, 0])
        error_cnt = np.array([0, 0, 0])
        for idx in range(8):
            ogn_index = [self.rec_indexes[0][idx], self.rec_indexes[1][idx], self.rec_indexes[2][idx]]  # original
            curr_index = [self.packet_header[self.packet_index][idx]] * 3
            curr_error = np.subtract(curr_index, ogn_index)
            for jdx in range(len(curr_error)):
                if abs(curr_error[jdx]) <= 1:
                    error_cnt[jdx] += 1
                else:
                    curr_error[jdx] = 0
            index_error = np.add(index_error, curr_error)
            self.corr_indexes[0].append(4*np.int16(ogn_index[0]/4))
            self.corr_indexes[1].append(4*np.int16(ogn_index[1]/4))
            self.corr_indexes[2].append(4*np.int16(ogn_index[2]/4))

        for idx in range(len(index_error)):
            if error_cnt[idx] == 0:
                index_error[idx] = 0
                error_cnt[idx] = 1
            index_error[idx] = np.rint(index_error[idx]/error_cnt[idx])

        if self.d_debug >= 2:
            print('Header Index Error: ', index_error)

        for idx in range(8, self.pld_chirps):
            ogn_index = [self.rec_indexes[0][idx], self.rec_indexes[1][idx], self.rec_indexes[2][idx]]
            # print(index_error)
            # print(np.mod(np.add(np.add(ogn_index, index_error), self.d_number_of_bins), self.d_number_of_bins))
            curr_index = np.int16(
                np.mod(np.add(np.add(ogn_index, index_error), self.d_number_of_bins), self.d_number_of_bins))
            self.corr_indexes[0].append(curr_index[0])
            self.corr_indexes[1].append(curr_index[1])
            self.corr_indexes[2].append(curr_index[2])

    def get_chirp_error(self):
        self.chirp_errors_list = np.abs(np.subtract(self.packet_chirp[self.packet_index], self.corr_indexes))
        for jdx in range(3):
            self.chirp_errors.append(np.count_nonzero(self.chirp_errors_list[jdx]))

    def recover_byte(self):
        for jdx in range(3):
            self.rec_bytes.append(self.lora_decoder.decode(self.corr_indexes[jdx].copy()))

    def get_byte_error(self):
        for idx in range(3):
            comp_result = []
            header_result = []
            for jdx in range(0, 3):
                header_result.append(
                    bin(self.packet_byte[self.packet_index][idx][jdx] ^ self.rec_bytes[idx][jdx]).count('1'))
            for jdx in range(0, self.packet_length):
                comp_result.append(
                    bin(self.packet_byte[self.packet_index][idx][jdx] ^ self.rec_bytes[idx][jdx]).count('1'))

            self.byte_errors.append(np.sum(comp_result))
            self.header_errors.append(np.sum(header_result))


    def save_result(self):
        curr_offsets = [self.chirp_offset, self.samp_offset, self.time_offset]
        self.detected_offsets.append(curr_offsets)
        curr_prebins = self.preamble_bins.copy()
        curr_sfdbins = self.sfd_bins.copy()
        curr_chirps = self.corr_indexes.copy()
        curr_bytes = self.rec_bytes.copy()
        curr_chirp_errors = self.chirp_errors.copy()
        curr_header_errors = self.header_errors.copy()
        curr_byte_errors = self.byte_errors.copy()

        for idx in range(3):
            self.decoded_prebins[idx].append(curr_prebins[idx])
            self.decoded_sfdbins[idx].append(curr_sfdbins[idx])
            self.decoded_chirps[idx].append(curr_chirps[idx])
            self.decoded_bytes[idx].append(curr_bytes[idx])
            self.decoded_chirp_errors[idx].append(curr_chirp_errors[idx])
            self.decoded_header_errors[idx].append(curr_header_errors[idx])
            self.decoded_byte_errors[idx].append(curr_byte_errors[idx])


    def show_result(self):
        if self.d_debug>=1:
            print("Show      result:")
            print("Chirp      Errors: ", self.chirp_errors)
            print("Header     Errors: ", self.header_errors)
            print("Byte       Errors: ", self.byte_errors)

    def clear_result(self):
        # clear all saved results
        self.detected_offsets.clear()

        for idx in range(3):
            self.decoded_prebins[idx].clear()
            self.decoded_sfdbins[idx].clear()
            self.decoded_chirps[idx].clear()
            self.decoded_bytes[idx].clear()
            self.decoded_chirp_errors[idx].clear()
            self.decoded_header_errors[idx].clear()
            self.decoded_byte_errors[idx].clear()


    def show_info(self):
        print('Show Info:')
        for idx in range(len(self.time_offset)):
            print('Chirp Offset {}: {}; Samp Offset {}: {}, Time Offset {}: {}'
                  .format(idx, self.chirp_offset[idx], idx, self.samp_offset[idx], idx, self.time_offset[idx]))

        print(self.packet_o.shape)

    def show_failed(self):
        print('Sync Failed:')
        for idx in range(len(self.time_offset)):
            print('Chirp Offset {}: {}; Samp Offset {}: {}, Time Offset {}: {}'
                  .format(idx, self.chirp_offset[idx], idx, self.samp_offset[idx], idx, self.time_offset[idx]))
        print('SFD Indexes: ', self.sfd_indexes)
        print('SFD bins:', self.sfd_bins)

def save_as_object(obj, name, sf, length, power):
    path_prefix = '../offResult/mlora3/'

    file_name = path_prefix + name +'_sf_' + str(sf) + '_len_' + str(length[0]) + '_pow_' + str(power[0]) + '_' + str(power[1]) + '_' + str(power[2]) + '.pkl'
    with open(file_name, 'wb') as output:
        pl.dump(obj, output, pl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    test_bw = 125e3
    test_samples_per_second = 1e6
    test_sf = 7
    test_cr = 4
    test_powers = [0, 0, -3]
    test_lengths = [16]* 3

    offresult3 = OffResult()

    mlora3 = MLoRa3(test_bw, test_samples_per_second, test_sf, test_cr, test_powers, test_lengths)
    mlora3.read_packets()
    mlora3.build_ideal_chirp()

    lora3 = LoRa3(test_bw, test_samples_per_second, test_sf, test_cr, test_powers, test_lengths)
    lora3.read_packets()
    lora3.build_ideal_chirp()

    for i in range(10):
        print("------------- {}-th for loop ---------------------:".format(i))
        for j in range(0, 100):
            mlora3.read_packet_shr(j)
            mlora3.init_decoding()
            mlora3.decoding_packet()

            lora3.read_packet_shr(j, mlora3.chirp_offset, mlora3.samp_offset, mlora3.packet_o)
            lora3.init_decoding()
            lora3.decoding_packet()

        offresult3.mlora_offsets.append(mlora3.detected_offsets.copy())
        offresult3.mlora_prebins.append([mlora3.decoded_prebins[j].copy() for j in range(3)])
        offresult3.mlora_sfdbins.append([mlora3.decoded_sfdbins[j].copy() for j in range(3)])
        offresult3.mlora_chirps.append([mlora3.decoded_chirps[j].copy() for j in range(3)])
        offresult3.mlora_bytes.append([mlora3.decoded_bytes[j].copy() for j in range(3)])
        offresult3.mlora_chirp_errors.append([mlora3.decoded_chirp_errors[j].copy() for j in range(3)])
        offresult3.mlora_header_errors.append([mlora3.decoded_header_errors[j].copy() for j in range(3)])
        offresult3.mlora_byte_errors.append([mlora3.decoded_byte_errors[j].copy() for j in range(3)])

        offresult3.lora_prebins.append(lora3.decoded_prebins.copy())
        offresult3.lora_sfdbins.append(lora3.decoded_sfdbins.copy())
        offresult3.lora_chirps.append(lora3.decoded_chirps.copy())
        offresult3.lora_bytes.append(lora3.decoded_bytes.copy())
        offresult3.lora_chirp_errors.append(lora3.decoded_chirp_errors.copy())
        offresult3.lora_header_errors.append(lora3.decoded_header_errors.copy())
        offresult3.lora_byte_errors.append(lora3.decoded_byte_errors.copy())

        mlora3.clear_result()
        lora3.clear_result()

        if mlora3.d_debug>=1:
            print("suc times", mlora3.sync_suc_cnt)
            print("fail times", mlora3.sync_fail_cnt)
            mlora3.sync_suc_cnt = 0
            mlora3.sync_fail_cnt = 0

    save_as_object(offresult3, 'offresult3', test_sf, test_lengths, test_powers)
