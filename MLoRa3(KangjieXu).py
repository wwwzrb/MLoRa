import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from cmath import phase
from scipy.fftpack import fft
from Stack import Stack, State
from lora_decode import LoRaDecode


class MLoRa3:

    def __init__(self, bw, samples_per_second, sf, cr, powers, lengths):
        self.fail_cnt = 0
        self.suc_cnt = 0
        # Debugging
        self.d_debug = 1

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

        self.packet_chirp = \
            [[[124, 0, 0, 76, 4, 108, 8, 120,
               1, 18, 67, 90, 66, 15, 31, 38,
               111, 2, 10, 108, 27, 42, 27, 33,
               50, 58, 80, 91, 32, 79, 57, 85],
              [124, 0, 0, 76, 4, 108, 8, 120,
               84, 71, 94, 85, 5, 122, 30, 123,
               58, 87, 49, 115, 106, 63, 24, 100,
               48, 68, 16, 123, 48, 71, 61, 87],
              [124, 0, 0, 76, 4, 108, 8, 120,
               1, 18, 67, 90, 66, 15, 31, 38,
               111, 2, 10, 108, 27, 42, 27, 33,
               50, 58, 80, 91, 32, 79, 57, 85]],
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
               60, 56, 45, 59, 47, 104, 33, 86],
              [56, 0, 28, 124, 116, 96, 108, 40,
               1, 18, 67, 90, 66, 15, 31, 38,
               111, 2, 10, 108, 27, 42, 27, 33,
               106, 17, 64, 80, 91, 109, 103, 18,
               1, 117, 83, 5, 53, 35, 119, 99,
               54, 66, 47, 69, 111, 64, 49, 92]]]

        self.packet_byte = np.array(
            [[[8, 128, 144, 18, 52, 86, 120, 18, 52, 86, 120, 0, 0, 0, 0, 0, 0, 0, 0],
              [8, 128, 144, 33, 67, 101, 135, 33, 67, 101, 135, 0, 0, 0, 0, 0, 0, 0, 0],
              [8, 128, 144, 18, 52, 86, 120, 18, 52, 86, 120, 0, 0, 0, 0, 0, 0, 0, 0]],
             [[16, 129, 64, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120],
              [16, 129, 64, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135],
              [16, 129, 64, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120]]],
            dtype=np.uint8)

        # Build standard chirp
        self.d_downchirp = []
        self.d_downchirp_zero = []
        self.d_upchirp = []
        self.d_upchirp_zero = []

        # Read overlapped packet
        self.chirp_cnt = None
        self.detect_scope = None
        self.chirp_offset = None
        self.samp_offset = None
        self.time_offset = None

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
        for i in range(3):
            self.pre_history.append(Stack())
            self.prebin_history.append(Stack())

        self.sfd_peak = 3
        self.sfd_chirps = 2
        self.sfd_tolerance = 4
        self.sfd_indexes = []
        self.sfd_bins = []
        self.sfd_begins = []
        self.sfd_history = []
        self.sfdbin_history = []
        self.sfdbins_history = []

        self.pld_peak = 3
        self.detect_offset_ab = None
        self.detect_offset_ac = None
        self.detect_chirp_offset_ac = None
        self.detect_chirp_offset_ab = None
        self.detect_samp_offset_ab = None
        self.detect_samp_offset_ac = None
        self.detect_bin_offset_ab = None
        self.detect_bin_offset_ac = None

        self.shift_samples = []
        self.pld_chirps = int(8 + (4 + self.d_cr) * np.ceil(self.d_lengths[0] * 2.0 / self.d_sf))
        self.pld_tolerance = 2
        self.pld_indexes = [[], [], []]
        self.pld_bins = [[], [], []]

        self.lora_decoder = LoRaDecode(self.payload_length, self.d_sf, self.d_cr)
        self.rec_indexes = [[], [], []]
        self.rec_bins = [[], [], []]
        self.oly_indexes = []
        self.corr_indexes = [[], [], []]
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_states = [State.S_RESET, State.S_RESET]

        # Save result
        self.detected_offsets = [[], [], []]
        self.decoded_prebins = [[], [], []]
        self.decoded_sfdbins = [[], [], []]
        self.decoded_chirps = [[], [], []]
        self.decoded_bytes = [[], [], []]
        self.decoded_chirp_errors = [[], [], []]
        self.decoded_header_errors = [[], [], []]
        self.decoded_byte_errors = [[], [], []]

        # MLORA 3 init
        self.buffer_length = 5  # 2*len(self.d_states) + 1


    def read_packets(self):
        path_prefix = "../dbgresult/FFT05051921/"
        exp_prefix = ["DIST/", "PWR/", "SF/", "PWR_LEN/"]

        power = self.d_powers
        length = self.d_lengths  # whats this???
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

        # 因为只有2个包的缘故
        self.packets = [data_0['packets'], data_1['packets'], data_0['packets']]
        self.packets_shr = [data_0['packets_shr'], data_1['packets_shr'], data_0['packets_shr']]

    def build_ideal_chirp(self):
        # 构造上下的标准chirp
        T = -0.5 * self.d_bw * self.d_symbols_per_second
        f0 = self.d_bw / 2.0
        pre_dir = 2.0 * np.pi * 1j
        cmx = np.complex(1.0, 1.0)  # ?? why exist

        # calculate exp chirp
        for idx in range(self.d_samples_per_symbol):
            t = self.d_dt * idx
            # T是负的，所以down乘1，up乘-1
            self.d_downchirp.append(np.exp(pre_dir * t * (f0 + T * t)))
            self.d_upchirp.append(np.exp(pre_dir * t * (f0 + T * t) * -1.0))

        # 为什么这么换一下????
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
        # self.chirp_offset = np.random.randint(0, self.chirp_cnt - 1)
        # self.chirp_offset1 = np.random.randint(0, self.chirp_cnt - 1)
        # self.chirp_offset2 = np.random.randint(self.chirp_offset1, self.chirp_cnt - 1)
        self.chirp_offset1 = 16
        self.chirp_offset2 = 30
        # self.samp_offset = self.d_decim_factor * np.random.randint(self.detect_scope + 1,
        #                       self.d_number_of_bins - self.detect_scope)
        self.bin_offset1 = np.random.randint(32, 48)
        self.samp_offset1 = self.bin_offset1 * self.d_decim_factor
        self.bin_offset2 = np.random.randint(72, 96)
        self.samp_offset2 = self.bin_offset2 * self.d_decim_factor

        self.time_offset1 = self.chirp_offset1 * self.d_samples_per_symbol + self.samp_offset1
        self.time_offset2 = self.chirp_offset2 * self.d_samples_per_symbol + self.samp_offset2

        self.packet = [self.packets[0][packet_idx], self.packets[1][packet_idx], self.packets[2][packet_idx]]
        self.packet_shr = [self.packets_shr[0][packet_idx], self.packets_shr[1][packet_idx],
                           self.packets_shr[2][packet_idx]]

        print(np.array(self.packet_shr).shape)
        print('TIME OFFSET1: {}'.format(self.time_offset1))
        print('TIME OFFSET2: {}'.format(self.time_offset2))

        # 叠加包
        # 包0 无重叠
        self.packet_o = self.packet_shr[0][:self.time_offset1]

        # 包1 2 重叠
        self.packet_o = np.concatenate((self.packet_o, np.add(self.packet_shr[0][self.time_offset1:self.time_offset2],
                                                              self.packet_shr[1][
                                                              :self.time_offset2 - self.time_offset1])))
        # 包1 2 3 重叠
        overlap_123_12 = np.add(self.packet_shr[0][self.time_offset2:],
                                self.packet_shr[1][self.time_offset2 - self.time_offset1:
                                                   len(self.packet_shr[1]) - self.time_offset1])
        overlap_123_123 = np.add(overlap_123_12, self.packet_shr[2][:len(self.packet_shr[2]) - self.time_offset2])
        self.packet_o = np.concatenate((self.packet_o, overlap_123_123))
        print(self.packet_o.shape)
        # 包2 3 重叠
        overlap_23 = np.add(self.packet_shr[2][len(self.packet_shr[2]) - self.time_offset2:
                                               len(self.packet_shr[2]) - (self.time_offset2 - self.time_offset1)],
                            self.packet_shr[1][len(self.packet_shr[1]) - self.time_offset1:])
        self.packet_o = np.concatenate((self.packet_o, overlap_23))
        print(self.packet_o.shape)
        # 包3 未重叠
        print(len(self.packet_shr[2]) - (self.time_offset2 - self.time_offset1))
        self.packet_o = np.concatenate((self.packet_o,
                                        self.packet_shr[2][len(self.packet_shr[2]) -
                                                           (self.time_offset2 - self.time_offset1):]))

        print(self.packet_o.shape)

    def init_decoding(self):
        self.bin_threshold = 9
        self.preamble_chirps = 7
        self.preamble_tolerance = 2
        self.preamble_indexes = []
        self.preamble_bins = []
        self.pre_history = []
        self.prebin_history = []
        for i in range(self.buffer_length):
            self.pre_history.append(Stack())
            self.prebin_history.append(Stack())

        self.sfd_chirps = 2
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
        self.pld_indexes = [[], [], []]
        self.pld_bins = [[], [], []]

        self.rec_indexes = [[], [], []]
        self.rec_bins = [[], [], []]
        self.oly_indexes = []
        self.corr_indexes = [[], [], []]
        self.rec_bytes = []
        self.chirp_errors = []
        self.header_errors = []
        self.byte_errors = []

        self.d_states = [State.S_RESET, State.S_RESET, State.S_RESET]
        for i in range(len(self.d_states)):
            if self.d_states[i] == State.S_RESET:
                for j in range(self.buffer_length):  # 会有5个峰
                    self.pre_history[j].clear()
                    self.prebin_history[j].clear()
                self.d_states[i] = State.S_PREFILL

    # 检测preamble位置
    def decoding_packet(self):

        print("---------------------------")
        print(" Decoding Packet")
        idx = 0
        while idx < len(self.packet_o):
            if idx + self.d_samples_per_symbol > len(self.packet_o):
                break

            bgn_index = idx
            end_index = idx + self.d_samples_per_symbol
            chirp_o = self.packet_o[bgn_index:end_index]
            chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp_o, self.d_samples_per_symbol, self.detect_scope,
                                                                  self.preamble_peak)

            for jdx in range(self.buffer_length):
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

                # case 2: 只有B 在preamble
                if self.d_states[0] != State.S_DETECT_PREAMBLE and self.d_states[1] == State.S_DETECT_PREAMBLE and \
                        self.d_states[2] != State.S_DETECT_PREAMBLE:
                    unique_index_b = np.where((detect_indexes != -1) & (detect_indexes != self.preamble_indexes[0]))[0]
                    repeat_index = np.where(detect_indexes != -1)[0]
                    if len(unique_index_b) > 0 and self.d_states[0] == State.S_SFD_SYNC:
                        self.d_states[1] = State.S_SFD_SYNC
                        self.preamble_indexes.append(detect_indexes[unique_index_b[0]])
                        self.preamble_bins.append(detect_bins[unique_index_b[0]])
                    if len(repeat_index) > 0 and self.d_states[0] != State.S_SFD_SYNC:
                        self.d_states[1] = State.S_SFD_SYNC
                        self.preamble_indexes.append(detect_indexes[repeat_index[0]])
                        self.preamble_bins.append(detect_bins[repeat_index[0]])

                # case 3: B C 同在preamble
                # [0]的preamble已经被检测出来了，进入了S_SFD_SYNC，另一个还在PREAMBLE
                if self.d_states[0] != State.S_DETECT_PREAMBLE and self.d_states[1] == State.S_DETECT_PREAMBLE and \
                        self.d_states[2] == State.S_DETECT_PREAMBLE:
                    # A 在SYNC
                    if self.d_states[0] == State.S_SFD_SYNC:
                        unique_index_b = \
                        np.where((detect_indexes != -1) & (detect_indexes != self.preamble_indexes[0]))[0]

                        if len(unique_index_b) > 0 and self.d_states[0] == State.S_SFD_SYNC:
                            self.d_states[1] = State.S_SFD_SYNC
                            self.preamble_indexes.append(detect_indexes[unique_index_b[0]])
                            self.preamble_bins.append(detect_bins[unique_index_b[0]])

                        if len(unique_index_b) > 1:
                            unique_index_c = \
                            np.where((detect_indexes != -1) & (detect_indexes != self.preamble_indexes[0]) &
                                     (detect_indexes != unique_index_b[1]))[0]
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
                    unique_index_c = np.where((detect_indexes != -1) & (detect_indexes != self.preamble_indexes[0]) &
                                              (detect_indexes != self.preamble_indexes[1]))[0]
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
        if len(self.sfd_begins) < 3:
            self.fail_cnt += 1
            return
        self.packet_decode(self.sfd_begins[0], self.sfd_begins[1], self.sfd_begins[2])
        return

        # self.save_result()
        # self.show_result()

    def get_fft_bins(self, content, window, scope, num):
        # ??? why multiply downchirp
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
                                         curr_fft[(
                                                          self.d_samples_per_symbol - self.d_half_number_of_bins):self.d_samples_per_symbol]))

        curr_fft = curr_decim_fft
        curr_index, curr_maxbin = self.get_max_bins(curr_fft, scope, num)

        return curr_index, curr_maxbin, curr_fft

    def error_correct(self):
        index_error = np.array([0, 0])
        for idx in range(8):
            ogn_index = [self.rec_indexes[0][idx], self.rec_indexes[1][idx]]  # original
            curr_index = np.int16(np.rint(np.divide(ogn_index, 4)))
            index_error = np.add(index_error, np.subtract(np.multiply(curr_index, 4), ogn_index))
            self.corr_indexes[0].append(4 * curr_index[0])
            self.corr_indexes[1].append(4 * curr_index[1])

        index_error = np.rint(np.divide(index_error, 8))
        for idx in range(8, self.pld_chirps):
            ogn_index = [self.rec_indexes[0][idx], self.rec_indexes[1][idx]]
            curr_index = np.int16(
                np.mod(np.add(np.add(ogn_index, index_error), self.d_number_of_bins), self.d_number_of_bins))
            self.corr_indexes[0].append(curr_index[0])
            self.corr_indexes[1].append(curr_index[1])

    def get_chirp_error(self):
        chirp_errors_list = np.abs(np.subtract(self.packet_chirp[self.packet_index], self.corr_indexes))
        self.chirp_errors.append(np.count_nonzero(chirp_errors_list[0]))
        self.chirp_errors.append(np.count_nonzero(chirp_errors_list[1]))

    def recover_byte(self):
        self.rec_bytes.append(self.lora_decoder.decode(self.corr_indexes[0].copy()))
        print(self.rec_bytes)
        self.rec_bytes.append(self.lora_decoder.decode(self.corr_indexes[1].copy()))
        print(self.rec_bytes)

    def get_byte_error(self):

        for idx in range(2):
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
        if len(self.sfd_begins) == 2:
            if abs(self.sfd_begins[1] - self.sfd_begins[
                0] - self.time_offset) < self.sfd_tolerance * self.d_decim_factor:
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

                print("Show      result:")
                self.show_result()
        else:
            print("Sync      failed!")

    def show_result(self):
        print("Chirp Number: ", self.chirp_cnt)
        print("Chirp Offset AB: ", self.detect_chirp_offset_ab)
        print("Chirp Offset AC: ", self.detect_chirp_offset_ac)
        print("Samp Offset AB: ", self.detect_samp_offset_ab)
        print("Samp Offset AC: ", self.detect_samp_offset_ac)
        print("Time Offset AB: ", self.detect_offset_ab)
        print("Time Offset AC: ", self.detect_offset_ac)

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
        print("Packet     chirp0: ", self.packet_chirp[0][0])
        print("CORR      indexe1: ", self.corr_indexes[1])
        print("Packet     chirp1: ", self.packet_chirp[0][1])
        print("Chirp      Errors: ", self.chirp_errors)
        print("Header     Errors: ", self.header_errors)
        print("Byte       Errors: ", self.byte_errors)

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

    def packet_decode(self, offa, offb, offc):

        print("SFD BEGINES:", offa, offb, offc)
        down_off = 2 * self.d_samples_per_symbol + self.d_quad_samples_per_symbol
        # 校准payload开始位置
        offa = offa + down_off  # - self.preamble_indexes[0] * self.d_decim_factor
        offb = offb + down_off  # - self.preamble_indexes[1] * self.d_decim_factor
        offc = offc + down_off  # - self.preamble_indexes[2] * self.d_decim_factor
        off12 = offb - offa
        off23 = offc - offb
        off13 = offc - offa

        # 获取chirp级别的offset
        chirp_off_ab = off12 // self.d_samples_per_symbol
        chirp_off_ac = off13 // self.d_samples_per_symbol

        # sample offset 先默认 ab的off小于bc的off
        sample_off_ab = off12 - chirp_off_ab * self.d_samples_per_symbol
        sample_off_ac = off13 - chirp_off_ac * self.d_samples_per_symbol

        # 先默认 soff1 < soff2吧
        soff1 = sample_off_ab
        soff2 = sample_off_ac
        sid = offa

        print("PAYLOAD BEGINS:", offa, offb, offc)
        print("CHIRP OFF AB: ", chirp_off_ab)
        print("CHIRP OFF AC: ", chirp_off_ac)
        print("SAMPLE OFF AB: ", soff1)
        print("SAMPLE OFF AC: ", soff2)

        self.detect_chirp_offset_ac = chirp_off_ac
        self.detect_chirp_offset_ab = chirp_off_ab
        self.detect_samp_offset_ab = sample_off_ab
        self.detect_samp_offset_ac = sample_off_ac
        self.detect_offset_ab = chirp_off_ab * self.d_samples_per_symbol + sample_off_ab
        self.detect_offset_ac = chirp_off_ac * self.d_samples_per_symbol + sample_off_ac

        if self.detect_offset_ab != self.time_offset1 or self.detect_offset_ac != self.time_offset2:
            print("Failed")
            self.fail_cnt += 1
        else:
            self.suc_cnt += 1
        return
        # PART 1:
        # 分别按A，B，C对齐，存入3个segment
        # free A
        for idx in range(0, chirp_off_ab):
            bgn_index = offa + idx * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:soff1] = chirp[0:soff1]
            chirp2[soff1:soff2] = chirp[soff1:soff2]
            chirp3[soff2:] = chirp[soff2:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp, self.d_samples_per_symbol,
                                                                  self.detect_scope, self.pld_peak)
            # 存入A中
            self.pld_indexes[0].append([chirp_index1, chirp_index2, chirp_index3, chirp_index])
            self.pld_bins[0].append([chirp_max1, chirp_max2, chirp_max3, chirp_max])

        # AB overlap
        for idx in range(chirp_off_ab, chirp_off_ac):
            # 按A对齐
            bgn_index = offa + idx * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:soff1] = chirp[0:soff1]
            chirp2[soff1:soff2] = chirp[soff1:soff2]
            chirp3[soff2:] = chirp[soff2:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            # 存入A中
            self.pld_indexes[0].append([chirp_index1, chirp_index2, chirp_index3])
            self.pld_bins[0].append([chirp_max1, chirp_max2, chirp_max3])

            # 按 B 对齐
            bgn_index = offb + (idx - chirp_off_ab) * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:soff2 - soff1] = chirp[0:soff2 - soff1]
            chirp2[soff2 - soff1:self.d_samples_per_symbol - soff1] = chirp[
                                                                      soff2 - soff1:self.d_samples_per_symbol - soff1]
            chirp3[self.d_samples_per_symbol - soff1:] = chirp[self.d_samples_per_symbol - soff1:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            # 存入B payload信息中
            self.pld_indexes[1].append([chirp_index1, chirp_index2, chirp_index3])
            self.pld_bins[1].append([chirp_max1, chirp_max2, chirp_max3])

        # ABC overlap
        for idx in range(chirp_off_ac, self.pld_chirps):
            # 按A对齐
            bgn_index = offa + idx * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:soff1] = chirp[0:soff1]
            chirp2[soff1:soff2] = chirp[soff1:soff2]
            chirp3[soff2:] = chirp[soff2:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            # 存入A中
            self.pld_indexes[0].append([chirp_index1, chirp_index2, chirp_index3])
            self.pld_bins[0].append([chirp_max1, chirp_max2, chirp_max3])

            # 按 B 对齐
            bgn_index = offb + (idx - chirp_off_ab) * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:soff2 - soff1] = chirp[0:soff2 - soff1]
            chirp2[soff2 - soff1:self.d_samples_per_symbol - soff1] = chirp[
                                                                      soff2 - soff1:self.d_samples_per_symbol - soff1]
            chirp3[self.d_samples_per_symbol - soff1:] = chirp[self.d_samples_per_symbol - soff1:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            # 存入B payload信息中
            self.pld_indexes[1].append([chirp_index1, chirp_index2, chirp_index3])
            self.pld_bins[1].append([chirp_max1, chirp_max2, chirp_max3])

            # 按 C 对齐
            bgn_index = offc + (idx - chirp_off_ac) * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:self.d_samples_per_symbol - soff2] = chirp[0:self.d_samples_per_symbol - soff2]
            chirp2[self.d_samples_per_symbol - soff2:self.d_samples_per_symbol - soff2 + soff1] = \
                chirp[self.d_samples_per_symbol - soff2:self.d_samples_per_symbol - soff2 + soff1]
            chirp3[self.d_samples_per_symbol - soff2 + soff1:] = chirp[self.d_samples_per_symbol - soff2 + soff1:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            # 存入C payload信息中
            self.pld_indexes[2].append([chirp_index1, chirp_index2, chirp_index3])
            self.pld_bins[2].append([chirp_max1, chirp_max2, chirp_max3])

        # B C overlap
        for idx in range(self.pld_chirps, self.pld_chirps + chirp_off_ab):
            # 按 B 对齐
            bgn_index = offb + (idx - chirp_off_ab) * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:soff2 - soff1] = chirp[0:soff2 - soff1]
            chirp2[soff2 - soff1:self.d_samples_per_symbol - soff1] = chirp[
                                                                      soff2 - soff1:self.d_samples_per_symbol - soff1]
            chirp3[self.d_samples_per_symbol - soff1:] = chirp[self.d_samples_per_symbol - soff1:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            # 存入B payload信息中
            self.pld_indexes[1].append([chirp_index1, chirp_index2, chirp_index3])
            self.pld_bins[1].append([chirp_max1, chirp_max2, chirp_max3])

            # 按 C 对齐
            bgn_index = offc + (idx - chirp_off_ac) * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:self.d_samples_per_symbol - soff2] = chirp[0:self.d_samples_per_symbol - soff2]
            chirp2[self.d_samples_per_symbol - soff2:self.d_samples_per_symbol - soff2 + soff1] = \
                chirp[self.d_samples_per_symbol - soff2:self.d_samples_per_symbol - soff2 + soff1]
            chirp3[self.d_samples_per_symbol - soff2 + soff1:] = chirp[self.d_samples_per_symbol - soff2 + soff1:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            # 存入C payload信息中
            self.pld_indexes[2].append([chirp_index1, chirp_index2, chirp_index3])
            self.pld_bins[2].append([chirp_max1, chirp_max2, chirp_max3])

        # free C
        for idx in range(self.pld_chirps+chirp_off_ab, self.pld_chirps + chirp_off_ac):
            # 按 C 对齐
            bgn_index = offc + (idx - chirp_off_ac) * self.d_samples_per_symbol
            end_index = bgn_index + self.d_samples_per_symbol
            chirp = self.packet_o[bgn_index:end_index]
            chirp1 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp2 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp3 = np.zeros(self.d_samples_per_symbol, dtype=np.complex)
            chirp1[0:self.d_samples_per_symbol - soff2] = chirp[0:self.d_samples_per_symbol - soff2]
            chirp2[self.d_samples_per_symbol - soff2:self.d_samples_per_symbol - soff2 + soff1] = \
                chirp[self.d_samples_per_symbol - soff2:self.d_samples_per_symbol - soff2 + soff1]
            chirp3[self.d_samples_per_symbol - soff2 + soff1:] = chirp[self.d_samples_per_symbol - soff2 + soff1:]
            chirp_index1, chirp_max1, chirp_bin1 = self.get_fft_bins(chirp1, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index2, chirp_max2, chirp_bin2 = self.get_fft_bins(chirp2, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index3, chirp_max3, chirp_bin3 = self.get_fft_bins(chirp3, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            chirp_index, chirp_max, chirp_bin = self.get_fft_bins(chirp, self.d_samples_per_symbol,
                                                                     self.detect_scope, self.pld_peak)
            # 存入C payload信息中
            self.pld_indexes[2].append([chirp_index1, chirp_index2, chirp_index3, chirp_index])
            self.pld_bins[2].append([chirp_max1, chirp_max2, chirp_max3, chirp_max])

        # PART 2:
        # symbol recovery

        # 排除后到的包的packet 对 先到的payload的 影响
        preamble_index_ab = (self.preamble_indexes[1] - self.preamble_indexes[0] +
                             self.d_quad_samples_per_symbol // self.d_decim_factor + self.d_number_of_bins) % self.d_number_of_bins
        preamble_index_ac = (self.preamble_indexes[2] - self.preamble_indexes[0] +
                             self.d_quad_samples_per_symbol // self.d_decim_factor + self.d_number_of_bins) % self.d_number_of_bins
        preamble_index_bc = (self.preamble_indexes[2] - self.preamble_indexes[1] +
                             self.d_quad_samples_per_symbol // self.d_decim_factor + self.d_number_of_bins) % self.d_number_of_bins
        print(preamble_index_ab, preamble_index_ac, preamble_index_bc)

        print("PAYLOAD CHIRPs Number", self.pld_chirps)
        print("PAYLOAD INDEX")
        for item in self.pld_indexes:
            print(type(item), len(item), item)
        print("-----")

        # symbol recovery A:
        # free A part
        # 这里是默认 chirp off ac < pld chirp
        for idx in range(0, max(chirp_off_ab - 12, 0)):
            # 未重叠部分直接加入
            self.rec_indexes[0].append(self.pld_indexes[0][idx][3][np.argmax(self.pld_bins[0][idx][3])])
            self.rec_bins[0].append(np.max(self.pld_bins[0][idx][3]))

        # A payload 与 B preamble 叠加，需要排除 B preamble index
        for idx in range(max(chirp_off_ab - 12, 0), min(chirp_off_ac - 12, chirp_off_ab)):
            curr_index_0 = []
            curr_max_0 = []
            for jdx in range(len(self.pld_indexes[0][idx][3])):
                # 排除与preamble B相同的可能
                if abs(self.pld_indexes[0][idx][3][jdx] - preamble_index_ab) >= self.pld_tolerance \
                        and self.pld_bins[0][idx][3][jdx] > self.bin_threshold:
                    curr_index_0.append(self.pld_indexes[0][idx][3][jdx])
                    curr_max_0.append(self.pld_bins[0][idx][3][jdx])
            if len(curr_index_0) >= 1:
                self.rec_indexes[0].append(curr_index_0[np.argmax(curr_max_0)])
                self.rec_bins[0].append(np.max(curr_max_0))
            else:
                # 若检测不到，可能是与preamble重叠了，剩下payload里面最大的肯定是preamble index
                self.rec_indexes[0].append(self.pld_indexes[0][idx][3][np.argmax(self.pld_bins[0][idx][3])])
                self.rec_bins[0].append(np.max(self.pld_bins[0][idx][3]))

        # A payload 与 B C preamble叠加，需要排除 B、C preamble index
        for idx in range(min(chirp_off_ac - 12, chirp_off_ab), chirp_off_ab):
            curr_index_0 = []
            curr_max_0 = []
            for jdx in range(len(self.pld_indexes[0][idx][3])):
                # 排除与preamble相同的可能
                if abs(self.pld_indexes[0][idx][3][jdx] - preamble_index_ab) >= self.pld_tolerance \
                        and self.pld_bins[0][idx][3][jdx] > self.bin_threshold \
                        and abs(self.pld_indexes[0][idx][3][jdx] - preamble_index_ac) >= self.pld_tolerance:
                    curr_index_0.append(self.pld_indexes[0][idx][3][jdx])
                    curr_max_0.append(self.pld_bins[0][idx][3][jdx])
            if len(curr_index_0) >= 1:
                self.rec_indexes[0].append(curr_index_0[np.argmax(curr_max_0)])
                self.rec_bins[0].append(np.max(curr_max_0))
            else:
                # 若检测不到，可能是与preamble B 或者 C 重叠了,
                # 但是是哪个呢？？？这里分不清楚
                self.rec_indexes[0].append(self.pld_indexes[0][idx][3][np.argmax(self.pld_bins[0][idx][3])])
                self.rec_bins[0].append(np.max(self.pld_bins[0][idx][3]))

        #  A与B的payload叠加,C包还未到达,
        for idx in range(chirp_off_ab, max(chirp_off_ab, chirp_off_ac - 12)):
            # 挑出3次重复的index
            # A中 存的3个segment的宽度
            # self.d_samples_per_symbol -> dsps
            # 0: [0, soff1]
            # 1: [soff1, soff2]
            # 2: [soff2, dsps]
            if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[0][idx][0]
                curr_max = self.pld_bins[0][idx][0]
            elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[0][idx][1]
                curr_max = self.pld_bins[0][idx][1]
            elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                curr_idx = self.pld_indexes[0][idx][2]
                curr_max = self.pld_bins[0][idx][2]
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[0][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[0][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[0][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3 three
            if len(rpt_index) == 1: # 理想情况
                print(curr_idx[rpt_index[0]])
                self.rec_indexes[0].append(curr_idx[rpt_index[0]])
                self.rec_bins[0].append(curr_max[rpt_index[0]])
            else:
                # 当A.B都满足cnt>3，则加入，后续cross decoding完成
                self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])

        # A的payload 与 B的payload 与 C的preamble 叠加
        # 这里是先找出3次重复，再剔除preamble index C
        for idx in range(max(chirp_off_ab, chirp_off_ac - 12), chirp_off_ac):
            # 先确定3次连续
            # A中 存的3个segment的宽度
            # self.d_samples_per_symbol -> dsps
            # 0: [0, soff1]
            # 1: [soff1, soff2]
            # 2: [soff2, dsps]
            if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[0][idx][2]
                curr_max = self.pld_bins[0][idx][2]
            elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[0][idx][0]
                curr_max = self.pld_bins[0][idx][0]
            elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                curr_idx = self.pld_indexes[0][idx][1]
                curr_max = self.pld_bins[0][idx][1]
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[0][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[0][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[0][idx][jdx][kdx])
            # 重复3次
            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3 three
            if len(rpt_index) == 1:
                # preamble 和 payload 杂在一起
                self.rec_indexes[0].append(curr_idx[rpt_index[0]])
                self.rec_bins[0].append(curr_max[rpt_index[0]])
            elif len(rpt_index) > 1:
                # 剔除 C preamble
                temp_idx = []
                temp_max = []
                for jdx in range(len(rpt_index)):
                    # 最后再排除 C preamble
                    if abs(curr_idx[rpt_index[jdx]] - preamble_index_ac) >= self.pld_tolerance:
                        temp_idx.append(curr_idx[rpt_index[jdx]])
                        temp_max.append(curr_max[rpt_index[jdx]])
                if len(temp_idx) == 1:
                    # 成功剔除preamble
                    self.rec_indexes[0].append(temp_idx[0])
                    self.rec_bins[0].append(temp_max[0])
                else:
                    self.rec_indexes[0].append(np.array(temp_idx))
                    self.rec_bins[0].append(np.array(temp_max))
            else:
                self.rec_indexes[0].append(np.array([]))
                self.rec_bins[0].append(np.array([]))
                # 没有3个的只能看2个了
                # self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                # self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

        # 解A所有其它的 A payload + B payload + C payload
        for idx in range(chirp_off_ac, self.pld_chirps):
            # A中 存的3个segment的宽度
            # self.d_samples_per_symbol -> dsps
            # 0: [0, soff1]
            # 1: [soff1, soff2]
            # 2: [soff2, dsps]
            if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[0][idx][0]
                curr_max = self.pld_bins[0][idx][0]
            elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[0][idx][1]
                curr_max = self.pld_bins[0][idx][1]
            elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                curr_idx = self.pld_indexes[0][idx][2]
                curr_max = self.pld_bins[0][idx][2]
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[0][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[0][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[0][idx][jdx][kdx])
            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3 three

            if len(rpt_index) == 1:
                self.rec_indexes[0].append(curr_idx[rpt_index[0]])
                self.rec_bins[0].append(curr_max[rpt_index[0]])
            else:
                self.rec_indexes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                self.rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])

        # 开始解B
        # B payload 与 A payload部分
        for idx in range(0, max(chirp_off_ac - chirp_off_ab - 12, 0)):
            # B中 存的3个segment的宽度
            # self.d_samples_per_symbol -> dsps
            # 0: [soff1, soff2]
            # 1: [soff2, dsps]
            # 2: [dsps, dsps+soff1]
            if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[1][idx][2]
                curr_max = self.pld_bins[1][idx][2]
            elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[1][idx][0]
                curr_max = self.pld_bins[1][idx][0]
            elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                curr_idx = self.pld_indexes[1][idx][1]
                curr_max = self.pld_bins[1][idx][1]
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            break
            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3 three
            if len(rpt_index) == 1:
                self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                self.rec_bins[1].append(curr_max[rpt_index[0]])
            else:
                # if len(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]]) == 1:
                #     self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]][0])
                #     self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]][0])
                # else:
                #     self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
                #     self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])
                self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])

        # 解 B与 A payload、C preamble部分
        for idx in range(max(chirp_off_ac-chirp_off_ab-12, 0), chirp_off_ac - chirp_off_ab):
            # B中 存的3个segment的宽度
            # self.d_samples_per_symbol -> dsps
            # 0: [soff1, soff2]
            # 1: [soff2, dsps]
            # 2: [dsps, dsps+soff1]
            if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[1][idx][2]
                curr_max = self.pld_bins[1][idx][2]
            elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[1][idx][0]
                curr_max = self.pld_bins[1][idx][0]
            elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                curr_idx = self.pld_indexes[1][idx][1]
                curr_max = self.pld_bins[1][idx][1]
            # curr_cnt存不包含preamble index的3次连续index
            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            # curr_cnt2用于存包含preamble index的3次连续index
            curr_cnt2 = np.zeros(len(curr_idx), dtype=np.int8)

            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance \
                                and abs(curr_idx[ldx] - preamble_index_bc) >= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            break
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt2[ldx] += 1
                            break
            rpt_index = np.where(curr_cnt >= 3)[0]  # repeat 3 three
            if len(rpt_index) == 0:
                # 没有则是preamble index，已存入curr_cnt2中
                rpt_index = np.where(curr_cnt2 >= 3)[0]
                if len(rpt_index) == 1:
                    self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                    self.rec_bins[1].append(curr_max[rpt_index[0]])
                else:
                    # 若加入preamble index 还是没有大于3的，怎么办？？？ 中间一段会出现
                    # 只能再排除preamble index，存一下大于2的index
                    # solution 1: 存入出现两次的，并排除preamble index
                    rpt_index = np.where(curr_cnt >= 2)[0]
                    self.rec_indexes[1].append(np.array(curr_idx)[rpt_index])
                    self.rec_bins[1].append(np.array(curr_max)[rpt_index])
                    # solution 2: 存入所有的
            elif len(rpt_index) == 1:
                self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                self.rec_bins[1].append(curr_max[rpt_index[0]])
            else:
                # 1个以上则全放入，之后则用cross validation解决
                self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])

        # B 的payload叠加部分 ABC，BC同时叠加，找三段连续
        for idx in range(chirp_off_ac - chirp_off_ab, self.pld_chirps):
            # B中 存的3个segment的宽度
            # self.d_samples_per_symbol -> dsps
            # 0: [soff1, soff2]
            # 1: [soff2, dsps]
            # 2: [dsps, dsps+soff1]
            if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[1][idx][2]
                curr_max = self.pld_bins[1][idx][2]
            elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[1][idx][0]
                curr_max = self.pld_bins[1][idx][0]
            elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                curr_idx = self.pld_indexes[1][idx][1]
                curr_max = self.pld_bins[1][idx][1]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[1][idx])):
                for kdx in range(len(curr_idx)):
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[1][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            break
                            # curr_idx[ldx].append(pld_indexes[1][idx][jdx][kdx])

            rpt_index = np.where(curr_cnt >= 3)[0]
            if len(rpt_index) == 1:
                self.rec_indexes[1].append(curr_idx[rpt_index[0]])
                self.rec_bins[1].append(curr_max[rpt_index[0]])
            else:
                # 这里可能存了两个，后面需要用cross validation解决；
                self.rec_indexes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                self.rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])

        # C的payload 与 A B payload 叠加
        for idx in range(self.pld_chirps - (chirp_off_ac - chirp_off_ab)):
            # 挑出宽度最大的 C存的顺序和A、B也不一样
            # self.d_samples_per_symbol -> dsps
            # 0: [soff2, dsps]
            # 1: [dsps, dsps+soff1]
            # 2: [dsps+soff1, dsps+soff2]
            if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[2][idx][1]
                curr_max = self.pld_bins[2][idx][1]
            elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                curr_idx = self.pld_indexes[2][idx][2]
                curr_max = self.pld_bins[2][idx][2]
            elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                curr_idx = self.pld_indexes[2][idx][0]
                curr_max = self.pld_bins[2][idx][0]

            curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
            for jdx in range(len(self.pld_indexes[2][idx])):
                for kdx in range(len(curr_idx)):
                    # print("i: {}, j: {}, k:{}, index:{}, bin:{}".format(idx, jdx, kdx,
                    #                                                     self.pld_indexes[2][idx][jdx][kdx],
                    #                                                     self.pld_bins[2][idx][jdx][kdx]))
                    for ldx in range(len(curr_idx)):
                        if abs(curr_idx[ldx] - self.pld_indexes[2][idx][jdx][kdx]) <= self.pld_tolerance:
                            curr_cnt[ldx] += 1
                            break
            rpt_index = np.where(curr_cnt == 3)[0]
            if len(rpt_index) == 1:
                self.rec_indexes[2].append(curr_idx[rpt_index[0]])
                self.rec_bins[2].append(curr_max[rpt_index[0]])
            else:
                # 这里可能存了两个，后面需要用cross validation解决；
                self.rec_indexes[2].append(np.array(curr_idx)[np.where(curr_cnt >= 3)[0]])
                self.rec_bins[2].append(np.array(curr_max)[np.where(curr_cnt >= 3)[0]])

        print(len(self.rec_indexes[2]))
        for idx in range(self.pld_chirps - (chirp_off_ac-chirp_off_ab), self.pld_chirps):
            self.rec_indexes[2].append(self.pld_indexes[2][idx][3][np.argmax(self.pld_bins[2][idx][3])])
            self.rec_bins[2].append(np.max(self.pld_bins[2][idx][3]))

        print("AFTER SYMBOL RECOVERY")
        for item in self.rec_indexes:
            print(type(item), len(item), item)
        print("------")
        # PART3:
        # cross decoding
        # 初始化参数
        self.detect_bin_offset_ab = sample_off_ab // self.d_decim_factor
        self.detect_bin_offset_ac = sample_off_ac // self.d_decim_factor
        self.detect_bin_offset_bc = self.detect_bin_offset_ac - self.detect_bin_offset_ab

        # 不同的packet ABC不相同
        # 提取 A
        for idx in range(chirp_off_ab + 1, self.pld_chirps):
            if not (type(self.rec_indexes[0][idx]) is int or type(self.rec_indexes[0][idx]) is np.int32):
                curr_idx = []
                oly_idx = []
                # 检测确定的b chirp
                for jdx in range(idx - chirp_off_ab - 1, idx - chirp_off_ab + 1):
                    if type(self.rec_indexes[1][jdx]) is int or type(self.rec_indexes[1][idx]) is np.int32:
                        # 注意减去bin offset
                        oly_idx.append((self.rec_indexes[1][jdx] - self.detect_bin_offset_ab + self.d_number_of_bins) % self.d_number_of_bins)
                # 检测确定的c chirp
                if idx >= chirp_off_ac + 1:
                    for jdx in range(idx - chirp_off_ac - 1, idx - chirp_off_ac + 1):
                        if type(self.rec_indexes[2][jdx]) is int or type(self.rec_indexes[2][idx]) is np.int32:
                            # 注意减去 ac 间bin offset
                            oly_idx.append((self.rec_indexes[2][jdx] - self.detect_bin_offset_ac
                                            + self.d_number_of_bins) % self.d_number_of_bins)
                # 排除其它chirp
                for jdx in range(len(self.rec_indexes[0][idx])):
                    is_oly = False  # overlay
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[0][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[0][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[0][idx] = curr_idx[0]
                else:
                    # 选出最宽的segment
                    if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                        jdx = 0
                    elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                        jdx = 1
                    elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                        jdx = 2
                    # 这个是之前用 detect down chirp的时候记录bin的大小选择
                    if self.sfd_bins[0] >= self.sfd_bins[1] and self.sfd_bins[0] >= self.sfd_bins[2]:
                        self.rec_indexes[0][idx] = self.pld_indexes[0][idx][jdx][0]
                        self.rec_bins[0][idx] = self.pld_bins[0][idx][jdx][0]
                    elif self.sfd_bins[1] >= self.sfd_bins[2] and self.sfd_bins[1] > self.sfd_bins[0]:
                        self.rec_indexes[0][idx] = self.pld_indexes[0][idx][jdx][1]
                        self.rec_bins[0][idx] = self.pld_bins[0][idx][jdx][1]
                    else:
                        self.rec_indexes[0][idx] = self.pld_indexes[0][idx][jdx][2]
                        self.rec_bins[0][idx] = self.pld_bins[0][idx][jdx][2]

        # cross decoding B
        for idx in range(0, self.pld_chirps):
            if not (type(self.rec_indexes[1][idx]) is int or type(self.rec_indexes[1][idx]) is np.int32):
                curr_idx = []
                oly_idx = []
                # 存入A中确定的chirp
                if idx + chirp_off_ab + 2 < self.pld_chirps:
                    for jdx in range(idx + chirp_off_ab, idx + chirp_off_ab + 2):
                        if type(self.rec_indexes[0][jdx]) is int or type(self.rec_indexes[0][idx]) is np.int32:
                            # 注意加上 bin offset
                            oly_idx.append((self.rec_indexes[0][jdx] + self.detect_bin_offset_ab +
                                            self.d_number_of_bins) % self.d_number_of_bins)
                # 存入C中确定的chirp
                if idx >= (chirp_off_ac - chirp_off_ab) + 1:
                    for jdx in range(idx - (chirp_off_ac - chirp_off_ab) - 1, idx - (chirp_off_ac - chirp_off_ab) + 1):
                        if type(self.rec_indexes[2][jdx]) is int or type(self.rec_indexes[2][idx]) is np.int32:
                            # 注意减去 bin offset
                            oly_idx.append((self.rec_indexes[2][jdx] - self.detect_bin_offset_bc +
                                            self.d_number_of_bins) % self.d_number_of_bins)
                # 排除
                for jdx in range(len(self.rec_indexes[1][idx])):
                    is_oly = False  # overlay
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[1][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[1][idx][jdx])

                if len(curr_idx) == 1:
                    self.rec_indexes[1][idx] = curr_idx[0]
                else:
                    # 选出最宽的segment
                    if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                        jdx = 0
                    elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                        jdx = 1
                    elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                        jdx = 2

                    # 这个是之前用 detect down chirp的时候记录bin的大小
                    # power mapping
                    if self.sfd_bins[0] >= self.sfd_bins[1] and self.sfd_bins[0] >= self.sfd_bins[2]:
                        self.rec_indexes[1][idx] = self.pld_indexes[1][idx][jdx][0]
                        self.rec_bins[1][idx] = self.pld_bins[1][idx][jdx][0]
                    elif self.sfd_bins[1] >= self.sfd_bins[2] and self.sfd_bins[1] > self.sfd_bins[0]:
                        self.rec_indexes[1][idx] = self.pld_indexes[1][idx][jdx][1]
                        self.rec_bins[1][idx] = self.pld_bins[1][idx][jdx][1]
                    else:
                        self.rec_indexes[1][idx] = self.pld_indexes[1][idx][jdx][2]
                        self.rec_bins[1][idx] = self.pld_bins[1][idx][jdx][2]

        # cross decoding C
        for idx in range(0, self.pld_chirps):
            if not (type(self.rec_indexes[2][idx]) is int or type(self.rec_indexes[2][idx]) is np.int32):
                curr_idx = []
                oly_idx = []
                # 排除A中确定chirp
                if idx + chirp_off_ac + 2 <= self.pld_chirps:
                    for jdx in range(idx + chirp_off_ac, idx + chirp_off_ac + 2):
                        if type(self.rec_indexes[0][jdx]) is int or type(self.rec_indexes[0][idx]) is np.int32:
                            # 注意加上bin offset
                            oly_idx.append((self.rec_indexes[0][jdx] + self.detect_bin_offset_ac +
                                            self.d_number_of_bins) % self.d_number_of_bins)
                # 排除B中确定chirp
                if idx + chirp_off_ac - chirp_off_ab + 2 <= self.pld_chirps:
                    for jdx in range(idx + chirp_off_ac - chirp_off_ab, idx + chirp_off_ac - chirp_off_ab + 2 ):
                        if type(self.rec_indexes[1][jdx]) is int or type(self.rec_indexes[1][idx]) is np.int32:
                            # 注意加上bin offset
                            oly_idx.append((self.rec_indexes[1][jdx] + self.detect_bin_offset_bc +
                                            self.d_number_of_bins) % self.d_number_of_bins)
                # 排除oly中index
                for jdx in range(len(self.rec_indexes[2][idx])):
                    is_oly = False  # overlay
                    for kdx in range(len(oly_idx)):
                        if abs(self.rec_indexes[2][idx][jdx] - oly_idx[kdx]) < self.pld_tolerance:
                            is_oly = True
                            break
                    if not is_oly:
                        curr_idx.append(self.rec_indexes[2][idx][jdx])
                if len(curr_idx) == 1:
                    self.rec_indexes[2][idx] = curr_idx[0]
                else:
                    print(curr_idx)
                    # 选出最宽的segment
                    if soff1 >= soff2 - soff1 and soff1 >= self.d_samples_per_symbol - soff2:
                        jdx = 0
                    elif soff2 - soff1 > soff1 and soff2 >= self.d_samples_per_symbol - soff2:
                        jdx = 1
                    elif self.d_samples_per_symbol - soff2 > soff1 and self.d_samples_per_symbol - soff2 > soff2 - soff1:
                        jdx = 2
                    # 这个是之前用 detect down chirp的时候记录bin的大小
                    # power mapping
                    if self.sfd_bins[0] >= self.sfd_bins[1] and self.sfd_bins[0] >= self.sfd_bins[2]:
                        self.rec_indexes[2][idx] = self.pld_indexes[2][idx][jdx][0]
                        self.rec_bins[2][idx] = self.pld_bins[2][idx][jdx][0]
                    elif self.sfd_bins[1] >= self.sfd_bins[2] and self.sfd_bins[1] > self.sfd_bins[0]:
                        self.rec_indexes[2][idx] = self.pld_indexes[2][idx][jdx][1]
                        self.rec_bins[2][idx] = self.pld_bins[2][idx][jdx][1]
                    else:
                        self.rec_indexes[2][idx] = self.pld_indexes[2][idx][jdx][2]
                        self.rec_bins[2][idx] = self.pld_bins[2][idx][jdx][2]

        # after cross decoding
        print("AFTER CROSS DECODING")
        for item in self.rec_indexes:
            print(type(item), len(item), item)
        print("-------")

        # error correct
        index_error = np.array([0, 0, 0])
        for idx in range(8):
            ogn_index = [self.rec_indexes[0][idx], self.rec_indexes[1][idx], self.rec_indexes[2][idx]]  # original
            curr_index = np.int16(np.rint(np.divide(ogn_index, 4)))
            index_error = np.add(index_error, np.subtract(np.multiply(curr_index, 4), ogn_index))
            self.corr_indexes[0].append(4 * curr_index[0])
            self.corr_indexes[1].append(4 * curr_index[1])
            self.corr_indexes[2].append(4 * curr_index[2])
        # 把误差统一减掉
        index_error = np.rint(np.divide(index_error, 8))
        for idx in range(8, self.pld_chirps):
            ogn_index = [self.rec_indexes[0][idx], self.rec_indexes[1][idx], self.rec_indexes[2][idx]]
            index_error = np.array(index_error, dtype=np.int32)
            print(index_error)
            print(np.mod(np.add(np.add(ogn_index, index_error), self.d_number_of_bins), self.d_number_of_bins))
            curr_index = np.int16(
                np.mod(np.add(np.add(ogn_index, index_error), self.d_number_of_bins), self.d_number_of_bins))
            self.corr_indexes[0].append(curr_index[0])
            self.corr_indexes[1].append(curr_index[1])
            self.corr_indexes[2].append(curr_index[2])

        # get chirp error
        chirp_errors_list = np.abs(np.subtract(self.packet_chirp[self.packet_index], self.corr_indexes))
        for j in range(3):
            self.chirp_errors.append(np.count_nonzero(chirp_errors_list[j]))

        # recover bytes
        for j in range(3):
            self.rec_bytes.append(self.lora_decoder.decode(self.corr_indexes[j].copy()))
            print(self.rec_bytes[j])

        # get byte error
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

            curr_offsets = [self.chirp_offset, self.samp_offset, self.time_offset]
            for idx in range(3):
                self.detected_offsets[idx].append(curr_offsets[idx])

        # save result
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

        print("Show      result:")
        self.show_result()
        self.suc_cnt += 1


if __name__ == "__main__":
    test_bw = 125e3
    test_samples_per_second = 1e6
    test_sf = 7
    test_cr = 4
    test_powers = [0, -3]
    test_lengths = [8, 8]

    mlora3 = MLoRa3(test_bw, test_samples_per_second, test_sf, test_cr, test_powers, test_lengths)
    mlora3.read_packets()
    mlora3.build_ideal_chirp()

    for i in range(100):
        print("------------- {}-th for loop ---------------------:".format(i))
        mlora3.read_packet_shr(i)
        mlora3.init_decoding()
        mlora3.decoding_packet()
    print("suc times", mlora3.suc_cnt)
    print("fail times", mlora3.fail_cnt)
