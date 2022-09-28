import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from cmath import phase
from scipy.signal import find_peaks


def show_signal(content, title):
    x = [i for i in range(len(content[0]))]
    plt.plot(x, content[0], 'k.--')
    plt.plot(x, content[1], 'r.--')
    plt.title(title)
    plt.show()


def build_ideal_chirp():
    T = -0.5*d_bw*d_symbols_per_second
    f0 = d_bw/2.0
    pre_dir = 2.0*np.pi*1j
    cmx = np.complex(1.0, 1.0)

    for i in range(d_samples_per_symbol):
        t = d_dt*i
        d_downchirp.append(np.exp(pre_dir*t*(f0+T*t)))
        d_upchirp.append(np.exp(pre_dir*t*(f0+T*t)*-1.0))


def build_phase_chirp(window):
    phase_shift = -np.pi/d_decim_factor
    accumulator = 0
    # self.std_down_chirp.append(-1+0j)
    # self.std_up_chirp.append(-1-0j)
    for i in range(2 * window):
        accumulator += phase_shift
        d_downchirp.append(np.conj(np.exp(1j * accumulator)))
        d_upchirp.append(np.exp(1j * accumulator))
        phase_shift += (2 * np.pi) / window / d_decim_factor


def max_gradient_idx(content):
    max_gradient = 0.1
    max_index = 0
    for i in range(1, d_number_of_bins):
        gradient = content[i-1] - content[i]
        if gradient > max_gradient:
            max_gradient = gradient
            max_index = i+1
    return (d_number_of_bins - max_index) % d_number_of_bins


def get_chirp_ifreq(content):
    curr_conj_mul = []
    for i in range(1, len(content)):
        curr_conj_mul.append(content[i]*np.conj(content[i-1]))
    curr_ifreq = [phase(conj_mul) for conj_mul in curr_conj_mul]
    curr_ifreq.append(curr_ifreq[-1])
    return curr_ifreq


def get_fft_bins(content, window, isheader):
    curr_mul = np.multiply(content, d_downchirp[:window])
    curr_fft = np.abs(fft(curr_mul))

    # curr_decim_fft = np.concatenate((curr_fft[:d_half_number_of_bins],
    #                                curr_fft[(d_samples_per_symbol-d_half_number_of_bins):d_samples_per_symbol]))

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
    curr_index, curr_max = get_max_bins(curr_fft)
    
    if isheader:
        curr_index = [(index[0]/4, index[1]/4) for index in curr_index]
        
    return curr_index, curr_max, curr_mul, curr_fft


def get_max_bins(content):
    curr_peaks = {}
    for i in range(len(content)):
        peak_found = True
        for j in range(-5, 6):
            if content[i] < content[(i+j) % len(content)]:
                peak_found = False
                break
        if peak_found:
            curr_peaks[i] = content[i]
        # if(content[i] > content[(i-1) % len(content)]) and (content[i] > content[(i+1) % len(content)]):
        #     curr_peaks[i] = content[i]
    while len(curr_peaks) < 3:
        curr_peaks.update({0: 0})
    sorted_peaks = sorted(curr_peaks.items(), key=lambda kv: kv[1])
    max_idx = [sorted_peaks[-1][0], sorted_peaks[-2][0], sorted_peaks[-3][0]]
    max_bin = [sorted_peaks[-1][1], sorted_peaks[-2][1], sorted_peaks[-3][1]]
    return max_idx, max_bin


# distance exp
# path_prefix = "../LoRaBeeExp02281240/"
# exp_prefix = ["DIST/", "PWR/", "SF/"]
#
# params_0 = "3m/data_"
# params_1 = "12m/data_"
# exp_idx = 0
# packet_idx = 0

# pwr exp
path_prefix = "../dbgresult/FFT04212128/"
exp_prefix = ["DIST/", "PWR/", "SF/"]

params_0 = "-3dBm/data_"
params_1 = "0dBm/data_"
exp_idx = 1
packet_idx = 0

# pwr len exp
# path_prefix = "../dbgresult/FFT04212128/"
# exp_prefix = ["DIST/", "PWR/", "SF/", "PWR_LEN/"]
#
# params_0 = "-3dBm-4byte/fwd/data_"
# params_1 = "-3dBm-4byte/inv/data_"
#
# exp_idx = 3
# packet_idx = 0
packet_0 = scipy.fromfile(path_prefix+exp_prefix[exp_idx]+params_0+str(packet_idx), scipy.complex64)
packet_1 = scipy.fromfile(path_prefix+exp_prefix[exp_idx]+params_1+str(packet_idx), scipy.complex64)

d_bw = 125e3
d_samples_per_second = 1e6
d_dt = 1.0/d_samples_per_second
d_downchirp = []
d_upchirp = []

d_sf = 7
d_decim_factor = int(d_samples_per_second/d_bw)
d_number_of_bins = (1 << d_sf)
d_symbols_per_second = d_bw/d_number_of_bins
d_samples_per_symbol = d_number_of_bins*d_decim_factor
d_half_number_of_bins = int(d_number_of_bins/2)
d_half_samples_per_symbol = int(d_samples_per_symbol/2)
chirp_cnt = int(len(packet_0)/d_samples_per_symbol)
chirp_index = np.array([124, 60, 96, 112, 24, 108, 48, 100,
                        1, 18, 67, 90, 66, 15, 31, 38,
                        0, 42, 117, 34, 7, 75, 114, 57], dtype=scipy.uint8)

# generate idea chirps
build_ideal_chirp()
# build_phase_chirp(d_samples_per_symbol)
d_downchirp_ifreq = get_chirp_ifreq(d_downchirp)
d_upchirp_ifreq = get_chirp_ifreq(d_upchirp)
show_signal([d_downchirp_ifreq, d_upchirp_ifreq], 'Std Chirp')

# decode collided packets with time offsets
# shift_half_symbol = np.random.randint(1, 8)
shift_half_symbol = 0.5
shift_samples = int(shift_half_symbol * d_half_samples_per_symbol)
packet_o = packet_0[:shift_samples]
packet_o = np.concatenate((packet_o, np.add(packet_0[shift_samples:], packet_1[:len(packet_1)-shift_samples])))
packet_o = np.concatenate((packet_o, packet_1[len(packet_1)-shift_samples:]))

fft_idx_0 = np.zeros([3, chirp_cnt], dtype=np.int8)
fft_idx_1 = np.zeros([3, chirp_cnt], dtype=np.int8)
ogn_idx_0 = np.zeros([3, chirp_cnt], dtype=np.int8)
ogn_idx_1 = np.zeros([3, chirp_cnt], dtype=np.int8)
fft_max_0 = np.zeros([3, chirp_cnt])
fft_max_1 = np.zeros([3, chirp_cnt])
chirp_bin = np.zeros([2, chirp_cnt, d_number_of_bins])
# chirp_bin = np.zeros([2, chirp_cnt, d_samples_per_symbol])
for i in range(min(0, chirp_cnt), min(8, chirp_cnt)):
    idx = i
    bgn_idx_0 = idx*d_samples_per_symbol
    end_idx_0 = (idx+1)*d_samples_per_symbol

    bgn_idx_1 = bgn_idx_0 + shift_samples
    end_idx_1 = end_idx_0 + shift_samples

    chirp_0 = packet_o[bgn_idx_0:end_idx_0]
    chirp_1 = packet_o[bgn_idx_1:end_idx_1]

    chirp_idx_0, chirp_max_0, chirp_mul_0, chirp_bin_0 = get_fft_bins(chirp_0, d_samples_per_symbol, False)
    chirp_idx_1, chirp_max_1, chirp_mul_1, chirp_bin_1 = get_fft_bins(chirp_1, d_samples_per_symbol, False)

    fft_idx_0[:, idx] = chirp_idx_0
    fft_max_0[:, idx] = chirp_max_0
    fft_idx_1[:, idx] = chirp_idx_1
    fft_max_1[:, idx] = chirp_max_1
    chirp_bin[:, idx, :] = [chirp_bin_0, chirp_bin_1]

    chirp_ogn_0 = packet_0[bgn_idx_0:end_idx_0]
    chirp_oly_0 = np.zeros(d_samples_per_symbol, dtype=np.complex)
    offset_0 = shift_samples % d_samples_per_symbol
    if (end_idx_0 - shift_samples) > 0:
        if (end_idx_0 - shift_samples) < d_samples_per_symbol:
            chirp_oly_0[offset_0:] = packet_1[bgn_idx_0:end_idx_0 - offset_0]
        else:
            chirp_oly_0 = packet_1[bgn_idx_0 - shift_samples:end_idx_0 - shift_samples]

    chirp_ogn_idx_0, chirp_ogn_max_0, chirp_ogn_mul_0, chirp_ogn_bin_0 \
        = get_fft_bins(chirp_ogn_0, d_samples_per_symbol, False)

    chirp_oly_idx_0, chirp_oly_max_0, chirp_oly_mul_0, chirp_oly_bin_0 \
        = get_fft_bins(chirp_oly_0, d_samples_per_symbol, False)

    ogn_idx_0[:, idx] = [chirp_ogn_idx_0[0], chirp_oly_idx_0[0], chirp_oly_idx_0[1]]

    chirp_ogn_ifreq_0 = get_chirp_ifreq(chirp_ogn_0)
    chirp_oly_ifreq_0 = get_chirp_ifreq(chirp_oly_0)

    # chirp_ogn_idx_0 = max_gradient_idx(
    #     np.add.reduceat(chirp_ogn_ifreq_0, np.arange(0, len(chirp_ogn_ifreq_0), d_decim_factor)))
    # chirp_oly_idx_0 = max_gradient_idx(
    #     np.add.reduceat(chirp_oly_ifreq_0, np.arange(0, len(chirp_oly_ifreq_0), d_decim_factor)))
    # ogn_idx_0[:, idx] = [chirp_ogn_idx_0, chirp_oly_idx_0]

    chirp_ogn_1 = packet_1[bgn_idx_0:end_idx_0]
    chirp_oly_1 = np.zeros(d_samples_per_symbol, dtype=np.complex)
    offset_1 = d_samples_per_symbol - offset_0
    if (bgn_idx_0 + shift_samples) < len(packet_0):
        if(len(packet_0) - bgn_idx_0) < d_samples_per_symbol:
            chirp_oly_1[0:offset_1] = packet_0[bgn_idx_0+offset_0:end_idx_0]
        else:
            chirp_oly_1 = packet_0[bgn_idx_0 + shift_samples:end_idx_0 + shift_samples]

    chirp_ogn_idx_1, chirp_ogn_max_1, chirp_ogn_mul_1, chirp_ogn_bin_1 \
        = get_fft_bins(chirp_ogn_1, d_samples_per_symbol, False)

    chirp_oly_idx_1, chirp_oly_max_1, chirp_oly_mul_1, chirp_oly_bin_1 \
        = get_fft_bins(chirp_oly_1, d_samples_per_symbol, False)

    ogn_idx_1[:, idx] = [chirp_ogn_idx_1[0], chirp_oly_idx_1[0], chirp_oly_idx_1[1]]

    chirp_ogn_ifreq_1 = get_chirp_ifreq(chirp_ogn_1)
    chirp_oly_ifreq_1 = get_chirp_ifreq(chirp_oly_1)

    # chirp_ogn_idx_1 = max_gradient_idx(
    #     np.add.reduceat(chirp_ogn_ifreq_1, np.arange(0, len(chirp_ogn_ifreq_1), d_decim_factor)))
    # chirp_oly_idx_1 = max_gradient_idx(
    #     np.add.reduceat(chirp_oly_ifreq_1, np.arange(0, len(chirp_oly_ifreq_1), d_decim_factor)))
    # ogn_idx_1[:, idx] = [chirp_ogn_idx_1, chirp_oly_idx_1]

    show_signal([chirp_bin_0, np.zeros(d_number_of_bins)], 'Oly FFT 0')
    show_signal([chirp_ogn_bin_0, chirp_oly_bin_0], 'Ogn FFT 0,1')
    show_signal([chirp_ogn_ifreq_0, chirp_oly_ifreq_0], 'Ogn ifreq 0')
    show_signal([chirp_bin_1, np.zeros(d_number_of_bins)], 'Oly FFT 1')
    show_signal([chirp_ogn_bin_1, chirp_oly_bin_1], 'Ogn FFT 1,0')
    show_signal([chirp_ogn_ifreq_1, chirp_oly_ifreq_1], 'Ogn ifreq 1')

# decoding two collision-free packets
# idx_0 = []
# idx_1 = []
# max_0 = []
# max_1 = []
# bin_0 = []
# bin_1 = []
# for i in range(min(8, chirp_cnt)):
#     idx = i
#     begin_index = idx*d_samples_per_symbol
#     end_index = (idx+1)*d_samples_per_symbol
#
#     chirp_0 = packet_0[begin_index:end_index]
#     chirp_1 = packet_1[begin_index:end_index]
#
#     chirp_idx_0, chirp_max_0, chirp_mul_0, chirp_bin_0 = get_fft_bins(chirp_0, d_samples_per_symbol, False)
#     chirp_idx_1, chirp_max_1, chirp_mul_1, chirp_bin_1 = get_fft_bins(chirp_1, d_samples_per_symbol, False)
#
#     chirp_ifreq_0 = get_chirp_ifreq(chirp_0)
#     chirp_ifreq_1 = get_chirp_ifreq(chirp_1)
#     show_signal([chirp_ifreq_0, chirp_ifreq_1])
#
#     idx_0.append(chirp_idx_0)
#     idx_1.append(chirp_idx_1)
#     max_0.append(chirp_max_0)
#     max_1.append(chirp_max_1)
#     bin_0.append(chirp_bin_0)
#     bin_1.append(chirp_bin_1)
#
#     show_signal([chirp_0, chirp_1])
#     show_signal([chirp_mul_0, chirp_mul_1])
#     show_signal([chirp_bin_0, chirp_bin_1])



