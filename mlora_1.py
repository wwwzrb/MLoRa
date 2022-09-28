import scipy
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from cmath import phase
from scipy.fftpack import fft
from Stack import Stack, State


def show_signal(content, title):
    x = [idx for idx in range(len(content[0]))]
    plt.plot(x, content[0], 'k.--')
    plt.plot(x, content[1], 'r.--')
    plt.title(title)
    plt.show()


def build_ideal_chirp():
    global d_samples_per_symbol
    global d_half_samples_per_symbol
    global d_downchirp
    global d_downchirp_zero
    global d_upchirp
    global d_upchirp_zero
    global d_bw

    T = -0.5*d_bw*d_symbols_per_second
    f0 = d_bw/2.0
    pre_dir = 2.0*np.pi*1j
    cmx = np.complex(1.0, 1.0)

    for idx in range(d_samples_per_symbol):
        t = d_dt*idx
        d_downchirp.append(np.exp(pre_dir*t*(f0+T*t)))
        d_upchirp.append(np.exp(pre_dir*t*(f0+T*t)*-1.0))

    d_downchirp_zero = np.concatenate((d_downchirp[d_half_samples_per_symbol:], d_downchirp[0:d_half_samples_per_symbol]))
    d_upchirp_zero = np.concatenate((d_upchirp[d_half_samples_per_symbol:], d_upchirp[0:d_half_samples_per_symbol]))


def get_chirp_ifreq(content):
    curr_conj_mul = []
    for idx in range(1, len(content)):
        curr_conj_mul.append(content[idx]*np.conj(content[idx-1]))
    curr_ifreq = [phase(conj_mul) for conj_mul in curr_conj_mul]
    curr_ifreq.append(curr_ifreq[-1])
    return curr_ifreq


def get_fft_bins(content, window, isheader, scope):
    global d_downchirp

    curr_mul = np.multiply(content, d_downchirp[:window])
    curr_fft = np.abs(fft(curr_mul))

    # only considering the previous half positive frequency bins and the last half negative frequency bins
    # in total (0~d_decim_factor*d_bw)
    # curr_decim_fft = np.concatenate((curr_fft[:d_half_number_of_bins],
    #                                  curr_fft[(d_samples_per_symbol-d_half_number_of_bins):d_samples_per_symbol]))

    # positive frequency bins
    # curr_decim_fft = curr_fft[:d_number_of_bins]

    # negative frequency bins
    # curr_decim_fft = curr_fft[d_samples_per_symbol - d_number_of_bins:]

    # sum of pos and neg frequency bins
    curr_decim_fft = np.add(curr_fft[:d_number_of_bins], curr_fft[d_samples_per_symbol - d_number_of_bins:])

    # curr_decim_fft = curr_fft

    curr_fft = curr_decim_fft
    # curr_max = np.amax(curr_fft)
    # curr_index = np.where((curr_max - curr_fft) < 0.1)[0]
    # curr_index = int(np.sum(curr_index) / len(curr_index))
    curr_index, curr_maxbin = get_max_bins(curr_fft, scope)

    if isheader:
        curr_index = [(index[0] / 4, index[1] / 4) for index in curr_index]

    return curr_index, curr_maxbin, curr_fft


def get_down_chirp_bin(content, scope):
    global d_upchirp_zero

    curr_mul = np.multiply(content, d_upchirp_zero)
    curr_fft = np.abs(fft(curr_mul))

    # only considering the previous half positive frequency bins and the last half negative frequency bins
    # in total (0~d_decim_factor*d_bw)
    curr_decim_fft = np.concatenate((curr_fft[:d_half_number_of_bins],
                                     curr_fft[(d_samples_per_symbol-d_half_number_of_bins):d_samples_per_symbol]))

    curr_fft = curr_decim_fft
    curr_index, curr_maxbin = get_max_bins(curr_fft, scope)

    return curr_index, curr_maxbin, curr_fft


def get_max_bins(content, scope):
    curr_peaks = {}
    for idx in range(len(content)):
        peak_found = True
        for jdx in range(-scope, scope+1):
            if content[idx] < content[(idx+jdx) % len(content)]:
                peak_found = False
                break
        if peak_found:
            curr_peaks[idx] = content[idx]
        # if(content[idx] > content[(idx-1) % len(content)]) and (content[idx] > content[(idx+1) % len(content)]):
        #     curr_peaks[idx] = content[idx]
    while len(curr_peaks) < 3:
        curr_peaks.update({0: 0})
    sorted_peaks = sorted(curr_peaks.items(), key=lambda kv: kv[1])
    max_idx = [sorted_peaks[-idx][0] for idx in range(1, 4)]
    max_bin = [sorted_peaks[-idx][1] for idx in range(1, 4)]
    # max_peaks = [(sorted_peaks[-idx][0], sorted_peaks[-idx][1]) for idx in range(3)]

    return max_idx, max_bin


def detect_preamble(idx_stacks, bin_stacks):
    global preamble_chirps
    global preamble_tolerance
    pre_idxes = np.full(len(idx_stacks), -1)
    pre_bins = np.full(len(idx_stacks), -1, dtype=float)
    for idx in range(len(idx_stacks)):
        pre_idx = idx_stacks[idx].bottom()
        pre_found = True

        for jdx in range(preamble_chirps):
            if abs(pre_idx - idx_stacks[idx].get_i(jdx)) >= preamble_tolerance or bin_stacks[idx].get_i(jdx) <= 10:
                pre_found = False
                break

        if pre_found:
            pre_idxes[idx] = pre_idx
            pre_bins[idx] = np.average(bin_stacks[idx].get_list())

    return pre_idxes, pre_bins


def detect_preamble_all(idx_stacks, bin_stacks):
    global preamble_chirps
    global preamble_tolerance
    pre_idxes = np.full(len(idx_stacks), -1)
    pre_bins = np.full(len(idx_stacks), -1, dtype=float)
    for idx in range(len(idx_stacks)):
        pre_idx = idx_stacks[idx].bottom()
        pre_found = True

        curr_idx = []
        curr_pos = []
        for jdx in range(preamble_chirps):
            curr_found = False
            for kdx in range(len(idx_stacks)):
                if abs(pre_idx - idx_stacks[kdx].get_i(jdx)) < preamble_tolerance and \
                        bin_stacks[kdx].get_i(jdx) > 10:
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
            pre_idxes[idx] = pre_idx
            pre_bins[idx] = np.average([bin_stacks[bin_idx[1]].get_i(bin_idx[0]) for bin_idx in curr_idx])

    return pre_idxes, pre_bins


def detect_downchirp(bgn_idx, scope):
    global packet_o
    global sfd_history
    global sfdbin_history
    global sfd_indexes
    global sfd_bins
    global d_half_number_of_bins
    global sfd_tolerance

    sfd_history.clear()
    sfdbin_history.clear()

    for idx in range(sfd_chirps):
        pad_chirp = np.zeros(d_samples_per_symbol, dtype=np.complex)
        curr_chirp = packet_o[bgn_idx+idx*d_samples_per_symbol:bgn_idx+(idx+1)*d_samples_per_symbol]
        pad_chirp[0:len(curr_chirp)] = curr_chirp
        sfd_idx, sfd_max, sfd_bin = get_down_chirp_bin(pad_chirp, scope)
        sfd_history.append(sfd_idx)
        sfdbin_history.append(sfd_max)

    sfd_found = True

    curr_idx = []
    for idx in range(sfd_chirps):
        curr_found = False
        for jdx in range(3):
            if abs(sfd_history[idx][jdx]-d_half_number_of_bins) <= sfd_tolerance and sfdbin_history[idx][jdx] != 0:
                curr_idx.append((idx, jdx))
                curr_found = True
                break

        if not curr_found:
            sfd_found = False
            break

    if sfd_found:
        sfd_indexes.append([sfd_history[bin_idx[0]][bin_idx[1]] for bin_idx in curr_idx])
        sfd_bins.append(np.sum([sfdbin_history[bin_idx[0]][bin_idx[1]] for bin_idx in curr_idx]))

        return True
    else:
        return False


def recover_packet():
    global preamble_indexes
    global d_samples_per_symbol
    global d_quad_samples_per_symbol
    global d_decim_factor
    global d_number_of_bins
    global detect_offset
    global sfd_bins
    global pld_chirps
    global pld_indexes
    global pld_bins
    global rec_indexes
    global rec_bins

    preamble_shift = (d_quad_samples_per_symbol - preamble_indexes[0] * d_decim_factor) // d_decim_factor
    preamble_index = (preamble_indexes[1] + preamble_shift + d_number_of_bins) % d_number_of_bins
    detect_chirp_offset = detect_offset // d_samples_per_symbol
    detect_samp_offset = detect_offset % d_samples_per_symbol
    detect_bin_offset = detect_samp_offset // d_decim_factor

    if sfd_bins[0] >= sfd_bins[1]:
        for jdx in range(0, max(detect_chirp_offset - 12, 0)):
            rec_indexes[0].append(pld_indexes[0][jdx][np.argmax(pld_bins[0][jdx])])
            rec_bins[0].append(np.max(pld_bins[0][jdx]))

        for jdx in range(max(detect_chirp_offset - 12, 0), detect_chirp_offset):
            curr_index_0 = []
            curr_max_0 = []
            for kdx in range(len(pld_indexes[0][jdx])):
                if abs(pld_indexes[0][jdx][kdx] - preamble_index) >= pld_tolerance:
                    curr_index_0.append(pld_indexes[0][jdx][kdx])
                    curr_max_0.append(pld_bins[0][jdx][kdx])

            if len(curr_index_0) >= 1:
                rec_indexes[0].append(curr_index_0[np.argmax(curr_max_0)])
                rec_bins[0].append(np.max(curr_max_0))
            else:
                rec_indexes[0].append(pld_indexes[0][jdx][np.argmax(pld_bins[0][jdx])])
                rec_bins[0].append(np.max(pld_bins[0][jdx]))

        for jdx in range(detect_chirp_offset, pld_chirps):
            rec_indexes[0].append(pld_indexes[0][jdx][np.argmax(pld_bins[0][jdx])])
            rec_bins[0].append(np.max(pld_bins[0][jdx]))

        for jdx in range(0, pld_chirps - detect_chirp_offset):
            curr_index_1 = []
            curr_max_1 = []
            if (jdx + detect_chirp_offset) < pld_chirps - 1:
                prev_index = pld_indexes[0][jdx + detect_chirp_offset][0]
                next_index = pld_indexes[0][jdx + detect_chirp_offset + 1][0]
            else:
                prev_index = pld_indexes[0][jdx + detect_chirp_offset][0]
                next_index = pld_indexes[0][jdx + detect_chirp_offset][0]

            oly_index = [prev_index, next_index]
            oly_index = np.mod(np.add(oly_index, detect_bin_offset), d_number_of_bins)
            print(oly_index)

            for kdx in range(len(pld_indexes[1][jdx])):
                is_oly = False
                for l in range(len(oly_index)):
                    if abs(pld_indexes[1][jdx][kdx] - oly_index[l]) < pld_tolerance:
                        is_oly = True
                        break
                if not is_oly:
                    curr_index_1.append(pld_indexes[1][jdx][kdx])
                    curr_max_1.append(pld_bins[1][jdx][kdx])

            if len(curr_index_1) >= 1:
                rec_indexes[1].append(curr_index_1[np.argmax(curr_max_1)])
                rec_bins[1].append(np.max(curr_max_1))
            else:
                rec_indexes[1].append(pld_indexes[1][jdx][np.argmax(pld_bins[1][jdx])])
                rec_bins[1].append(np.max(pld_bins[1][jdx]))

        for jdx in range(pld_chirps - chirp_offset, pld_chirps):
            rec_indexes[1].append(pld_indexes[1][jdx][np.argmax(pld_bins[1][jdx])])
            rec_bins[1].append(np.max(pld_bins[1][jdx]))


def error_correct():
    global pld_chirps
    global rec_indexes
    global corr_indexes

    index_error = np.array([0, 0])
    for idx in range(8):
        ogn_index = [rec_indexes[0][idx], rec_indexes[1][idx]]
        curr_index = np.int8(np.rint(np.divide(ogn_index, 4)))
        index_error = np.add(index_error, np.subtract(np.multiply(curr_index, 4), ogn_index))
        corr_indexes[0].append(curr_index[0])
        corr_indexes[1].append(curr_index[1])

    index_error = np.rint(np.divide(index_error, 8))
    for idx in range(8, pld_chirps):
        ogn_index = [rec_indexes[0][idx], rec_indexes[1][idx]]
        curr_index = np.int8(np.mod(np.add(np.add(ogn_index, index_error), d_number_of_bins), d_number_of_bins))
        corr_indexes[0].append(curr_index[0])
        corr_indexes[1].append(curr_index[1])


def get_chirp_error():
    global packet_chirp
    global corr_indexes
    global chirp_errors

    chirp_errors = np.sum(np.abs(np.subtract(packet_chirp, corr_indexes)), axis=1)


def show_chirp(bgn_index, end_index, packet_index):
    global packet_shr
    global detect_offset
    chirp_ogn = np.zeros(d_samples_per_symbol, dtype=np.complex)

    chirp_oly = np.zeros(d_samples_per_symbol, dtype=np.complex)
    offset = detect_offset % d_samples_per_symbol

    if packet_index == 0:
        chirp_ogn[0:len(packet_shr[0][bgn_index:end_index])] = packet_shr[0][bgn_index:end_index]
        if (end_index - detect_offset) > 0:
            if (end_index - detect_offset) < d_samples_per_symbol:
                chirp_oly[offset:] = \
                    packet_shr[1][bgn_index - detect_offset + offset:end_index - detect_offset]
            else:
                chirp_oly = packet_shr[1][bgn_index - detect_offset:end_index - detect_offset]
    else:
        chirp_ogn[0:len(packet_shr[1][bgn_index - detect_offset:end_index - detect_offset])] = \
            packet_shr[1][bgn_index-detect_offset:end_index-detect_offset]
        if bgn_index < len(packet_shr[0]):
            if (len(packet_shr[0]) - bgn_index) < d_samples_per_symbol:
                chirp_oly[0:d_samples_per_symbol-offset] = \
                    packet_shr[0][bgn_index:end_index - offset]
            else:
                chirp_oly[0:len(packet_shr[0][bgn_index:end_index])] = packet_shr[0][bgn_index:end_index]

    chirp_ogn_index, chirp_ogn_max, chirp_ogn_bin = \
        get_fft_bins(chirp_ogn, d_samples_per_symbol, False, 2)

    chirp_oly_index, chirp_oly_max, chirp_oly_bin = \
        get_fft_bins(chirp_oly, d_samples_per_symbol, False, 2)

    chirp_ogn_ifreq = get_chirp_ifreq(chirp_ogn)
    chirp_oly_ifreq = get_chirp_ifreq(chirp_oly)

    show_signal([chirp_ogn_bin, chirp_oly_bin], 'Ogn FFT 0,1')
    show_signal([chirp_ogn_ifreq, chirp_oly_ifreq], 'Ogn ifreq 0,1')


path_prefix = "../dbgresult/FFT04212128/"
exp_prefix = ["DIST/", "PWR/", "SF/", "PWR_LEN/"]

power = [0, -3]
length = [4, 4]
payload = ['fwd', 'inv']

exp_idx = 3
packet_idx = 25

file_name = [path_prefix + exp_prefix[exp_idx] + "data_" + payload[0] + "_sf_7_len_" + str(length[0]) + "_pwr_" + str(power[0]) + ".mat",
             path_prefix + exp_prefix[exp_idx] + "data_" + payload[1] + "_sf_7_len_" + str(length[1]) + "_pwr_" + str(power[1]) + ".mat"]

data_0 = sio.loadmat(file_name[0])
data_1 = sio.loadmat(file_name[1])

packets = [data_0['packets'], data_1['packets']]
packets_shr = [data_0['packets_shr'], data_1['packets_shr']]

packet = [packets[0][packet_idx], packets[1][packet_idx]]
packet_shr = [packets_shr[0][packet_idx], packets_shr[1][packet_idx]]

packet_chirp = [[31, 15,  24,  28,  6,  27,  12,  25,
                 1, 18,  67,  90, 66,  15,  31,  38, 0, 42, 117,  34,  7,  75, 114, 57],
                [31, 15,  24,  28,  6,  27,  12,  25,
                 84, 18, 120, 106, 102, 68, 76, 30, 1, 42,  74,  34,   7,  76, 113, 57]]

d_bw = 125e3
d_samples_per_second = 1e6
d_dt = 1.0/d_samples_per_second
d_downchirp = []
d_downchirp_zero = []
d_upchirp = []
d_upchirp_zero = []

d_sf = 7
d_cr = 4
d_decim_factor = int(d_samples_per_second/d_bw)
d_number_of_bins = (1 << d_sf)
d_symbols_per_second = d_bw/d_number_of_bins
d_samples_per_symbol = d_number_of_bins*d_decim_factor
d_half_number_of_bins = int(d_number_of_bins/2)
d_half_samples_per_symbol = int(d_samples_per_symbol/2)
d_quad_samples_per_symbol = int(d_samples_per_symbol/4)
chirp_cnt = int(len(packet_shr[0])/d_samples_per_symbol)

# generate idea chirps
build_ideal_chirp()
# build_phase_chirp(d_samples_per_symbol)
d_downchirp_ifreq = get_chirp_ifreq(d_downchirp)
d_upchirp_ifreq = get_chirp_ifreq(d_upchirp)
# show_signal([d_downchirp_ifreq, d_upchirp_ifreq], 'Std Chirp')

# all time offset space
# chirp_offset = np.random.randint(0, chirp_cnt)
# samp_offset = np.random.randint(0, d_samples_per_symbol)
# time_offset = chirp_offset * d_samples_per_symbol + samp_offset

# controlled time offset space
preamble_scope = 2
# chirp_offset = np.random.randint(0, chirp_cnt-1)
chirp_offset = 14
samp_offset = np.random.randint(preamble_scope+1, d_number_of_bins-preamble_scope) * d_decim_factor
# samp_offset = 70*d_decim_factor

time_offset = chirp_offset * d_samples_per_symbol + samp_offset

print("Chirp Offset: ", chirp_offset)
print("Samp Offset: ", samp_offset)
print("Time Offset: ", time_offset)

packet_o = packet_shr[0][:time_offset]
packet_o = np.concatenate((packet_o, np.add(packet_shr[0][time_offset:], packet_shr[1][:len(packet_shr[1])-time_offset])))
packet_o = np.concatenate((packet_o, packet_shr[1][len(packet_shr[1])-time_offset:]))

preamble_chirps = 6
preamble_tolerance = 2
preamble_indexes = []
preamble_bins = []
index_history = []
maxbin_history = []
for i in range(3):
    index_history.append(Stack())
    maxbin_history.append(Stack())

sfd_chirps = 2
sfd_tolerance = 2
sfd_indexes = []
sfd_bins = []
sfd_begins = []
sfd_history = []
sfdbin_history = []

detect_offset = time_offset
shift_samples = []
pld_chirps = int(8 + (4+d_cr) * np.ceil(length[0]*2.0/d_sf))
pld_tolerance = 2
pld_indexes = [[], []]
pld_bins = [[], []]

rec_indexes = [[], []]
rec_bins = [[], []]
corr_indexes = [[], []]
chirp_errors = []

d_states = [State.S_RESET, State.S_RESET]
for i in range(len(d_states)):
    if d_states[i] == State.S_RESET:
        for j in range(3):
            index_history[j].clear()
            maxbin_history[j].clear()
        d_states[i] = State.S_PREFILL

i = 0
while i < (len(packet_o)):
    if i+d_samples_per_symbol > len(packet_o):
        break

    bgn_index_0 = i
    end_index_0 = i + d_samples_per_symbol
    chirp_o = packet_o[i:i + d_samples_per_symbol]
    chirp_index, chirp_max, chirp_bin = get_fft_bins(chirp_o, d_samples_per_symbol, False, 2)

    # if i//d_samples_per_symbol >= chirp_offset:
    #     show_signal([chirp_bin, np.zeros(d_number_of_bins)],
    #                 'Oly FFT 0 with index ' + str(i//d_samples_per_symbol))
    #     show_chirp(bgn_index_0, end_index_0, 0)

    for j in range(3):
        index_history[j].push(chirp_index[j])
        maxbin_history[j].push(chirp_max[j])

        if index_history[j].size() > preamble_chirps:
            index_history[j].pop_back()
            maxbin_history[j].pop_back()

    if d_states[0] == State.S_PREFILL and d_states[1] == State.S_PREFILL:
        if index_history[0].size() >= preamble_chirps:
            for j in range(2):
                d_states[j] = State.S_DETECT_PREAMBLE
        else:
            i += d_samples_per_symbol
            continue

    if d_states[0] == State.S_DETECT_PREAMBLE or d_states[1] == State.S_DETECT_PREAMBLE:
        detect_indexes, detect_bins = detect_preamble_all(index_history, maxbin_history)
        # detect_indexes, detect_bins = detect_preamble(pre_history, maxbin_history)
        if d_states[0] == State.S_DETECT_PREAMBLE and d_states[1] == State.S_DETECT_PREAMBLE:
            if len(np.where(detect_indexes != -1)[0]) == 0:
                i += d_samples_per_symbol
                continue
            else:
                repeat_index = np.where(detect_indexes != -1)[0]
                if len(repeat_index) >= 2:
                    for j in range(2):
                        d_states[j] = State.S_SFD_SYNC
                        preamble_indexes.append(detect_indexes[repeat_index[j]])
                        preamble_bins.append(detect_bins[repeat_index[j]])

                    # align with the first packet
                    # i -= preamble_indexes[0] * d_decim_factor
                    # preamble_indexes[1] += preamble_indexes[0]
                else:
                    d_states[0] = State.S_SFD_SYNC
                    preamble_indexes.append(detect_indexes[repeat_index[0]])
                    preamble_bins.append(detect_bins[repeat_index[0]])
                    # align with the first packet
                    # i -= preamble_indexes[0] * d_decim_factor

                i += d_samples_per_symbol
                continue

        if d_states[0] != State.S_DETECT_PREAMBLE and d_states[1] == State.S_DETECT_PREAMBLE:
            unique_index = np.where((detect_indexes != -1) & (detect_indexes != preamble_indexes[0]))[0]
            repeat_index = np.where(detect_indexes != -1)[0]
            if len(unique_index) > 0 and d_states[0] == State.S_SFD_SYNC:
                d_states[1] = State.S_SFD_SYNC
                preamble_indexes.append(detect_indexes[unique_index[0]])
                preamble_bins.append(detect_bins[unique_index[0]])

            if len(repeat_index) > 0 and d_states[0] != State.S_SFD_SYNC:
                d_states[1] = State.S_SFD_SYNC
                preamble_indexes.append(detect_indexes[repeat_index[0]])
                preamble_bins.append(detect_bins[repeat_index[0]])

    if d_states[0] == State.S_SFD_SYNC or d_states[1] == State.S_SFD_SYNC:

        if d_states[0] == State.S_SFD_SYNC and d_states[1] != State.S_SFD_SYNC:
            bgn_sfd = i - preamble_indexes[0] * d_decim_factor
            if detect_downchirp(bgn_sfd, 2):
                d_states[0] = State.S_READ_PAYLOAD
                # i += 2*d_samples_per_symbol + d_quad_samples_per_symbol - preamble_indexes[0] * d_decim_factor
                # Record shift samples needed to align with the first packet
                shift_samples.append(2*d_samples_per_symbol + d_quad_samples_per_symbol
                                     - preamble_indexes[0] * d_decim_factor)
                sfd_begins.append(bgn_sfd)

        if d_states[0] != State.S_SFD_SYNC and d_states[1] == State.S_SFD_SYNC:
            bgn_sfd = i - preamble_indexes[1] * d_decim_factor
            if detect_downchirp(bgn_sfd, 2):
                d_states[1] = State.S_READ_PAYLOAD
                # Record shift samples need to align with the second packet
                shift_samples.append(2*d_samples_per_symbol + d_quad_samples_per_symbol
                                     - preamble_indexes[1] * d_decim_factor)
                sfd_begins.append(bgn_sfd)

        if d_states[0] == State.S_SFD_SYNC and d_states[1] == State.S_SFD_SYNC:
            bgn_sfd = i - preamble_indexes[0] * d_decim_factor
            if detect_downchirp(bgn_sfd, 2):
                d_states[0] = State.S_READ_PAYLOAD
                # i += 2*d_samples_per_symbol + d_quad_samples_per_symbol - preamble_indexes[0] * d_decim_factor
                # Record shift samples needed to align with the first packet
                shift_samples.append(2*d_samples_per_symbol + d_quad_samples_per_symbol
                                     - preamble_indexes[0] * d_decim_factor)
                sfd_begins.append(bgn_sfd)

            bgn_sfd = i - preamble_indexes[1] * d_decim_factor
            if detect_downchirp(bgn_sfd, 2):
                d_states[1] = State.S_READ_PAYLOAD
                # Record shift samples need to align with the second packet
                shift_samples.append(2*d_samples_per_symbol + d_quad_samples_per_symbol
                                     - preamble_indexes[1] * d_decim_factor)
                sfd_begins.append(bgn_sfd)

    if d_states[0] == State.S_READ_PAYLOAD or d_states[1] == State.S_READ_PAYLOAD:

        if d_states[0] == State.S_READ_PAYLOAD and d_states[1] != State.S_READ_PAYLOAD:

            if len(pld_indexes[0]) < pld_chirps:
                chirp_0 = packet_o[i + shift_samples[0]:i + shift_samples[0] + d_samples_per_symbol]
                chirp_index_0, chirp_max_0, chirp_bin_0 = get_fft_bins(chirp_0, d_samples_per_symbol, False, 2)

                # chirp_index_0 = [chirp_index_0[j] for j in np.where(np.array(chirp_max_0) > 10)[0]]
                # chirp_max_0 = [chirp_max_0[j] for j in np.where(np.array(chirp_max_0) > 10)[0]]

                pld_indexes[0].append(chirp_index_0)
                pld_bins[0].append(chirp_max_0)

            if len(pld_indexes[0]) >= pld_chirps:
                d_states[0] = State.S_STOP

        if d_states[0] != State.S_READ_PAYLOAD and d_states[1] == State.S_READ_PAYLOAD:

            if len(pld_indexes[1]) < pld_chirps:
                chirp_1 = packet_o[i + shift_samples[1]:i + shift_samples[1] + d_samples_per_symbol]
                chirp_index_1, chirp_max_1, chirp_bin_1 = get_fft_bins(chirp_1, d_samples_per_symbol, False, 2)

                # chirp_index_1 = [chirp_index_1[j] for j in np.where(np.array(chirp_max_1) > 10)[0]]
                # chirp_max_1 = [chirp_max_1[j] for j in np.where(np.array(chirp_max_1) > 10)[0]]

                pld_indexes[1].append(chirp_index_1)
                pld_bins[1].append(chirp_max_1)

            if len(pld_indexes[1]) >= pld_chirps:
                d_states[1] = State.S_STOP

        if d_states[0] == State.S_READ_PAYLOAD and d_states[1] == State.S_READ_PAYLOAD:
            detect_offset = sfd_begins[1] - sfd_begins[0]

            if len(pld_indexes[0]) < pld_chirps:
                chirp_0 = packet_o[i + shift_samples[0]:i + shift_samples[0] + d_samples_per_symbol]
                chirp_index_0, chirp_max_0, chirp_bin_0 = get_fft_bins(chirp_0, d_samples_per_symbol, False, 2)

                # chirp_index_0 = [chirp_index_0[j] for j in np.where(np.array(chirp_max_0) > 10)[0]]
                # chirp_max_0 = [chirp_max_0[j] for j in np.where(np.array(chirp_max_0) > 10)[0]]

                pld_indexes[0].append(chirp_index_0)
                pld_bins[0].append(chirp_max_0)

            if len(pld_indexes[0]) >= pld_chirps:
                d_states[0] = State.S_STOP

            if len(pld_indexes[1]) < pld_chirps:
                chirp_1 = packet_o[i + shift_samples[1]:i + shift_samples[1] + d_samples_per_symbol]
                chirp_index_1, chirp_max_1, chirp_bin_1 = get_fft_bins(chirp_1, d_samples_per_symbol, False, 2)

                # chirp_index_1 = [chirp_index_1[j] for j in np.where(np.array(chirp_max_1) > 10)[0]]
                # chirp_max_1 = [chirp_max_1[j] for j in np.where(np.array(chirp_max_1) > 10)[0]]

                pld_indexes[1].append(chirp_index_1)
                pld_bins[1].append(chirp_max_1)

            if len(pld_indexes[1]) >= pld_chirps:
                d_states[1] = State.S_STOP

    if d_states[0] == State.S_STOP and d_states[1] == State.S_STOP:
        recover_packet()
        error_correct()
        get_chirp_error()

        print("Preamble  successful!")
        print("Preamble  indexes: ", preamble_indexes)
        print("Preamble  bins   : ", preamble_bins)
        print("SFD       indexes: ", sfd_indexes)
        print("SFD       bins   : ", sfd_bins)
        print("SFD       begins : ", sfd_begins)
        print("Downchirp successful!")
        print("PLD       indexe0: ", pld_indexes[0])
        print("PLD       indexe1: ", pld_indexes[1])
        print("PLD       bins0  : ", pld_bins[0])
        print("PLD       bins1  : ", pld_bins[1])
        print("REC       indexe0: ", rec_indexes[0])
        print("REC       indexe1: ", rec_indexes[1])
        print("REC       bins0  : ", rec_bins[0])
        print("REC       bins1  : ", rec_bins[1])
        print("CORR      indexe0: ", corr_indexes[0])
        print("Packet     chirp0: ", packet_chirp[0])
        print("CORR      indexe1: ", corr_indexes[1])
        print("Packet     chirp1: ", packet_chirp[1])
        print("Chirp      Errors: ", chirp_errors)

        break
    else:
        i += d_samples_per_symbol
        continue



