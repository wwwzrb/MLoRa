import numpy as np
import scipy
import matplotlib.pyplot as plt
from cmath import phase
from scipy.fftpack import fft
from Stack import State
from lora_decode import LoRaDecode


def show_signal(content, title):
    x = [idx for idx in range(len(content))]
    plt.plot(x, content, 'k.--')
    plt.title(title)
    plt.show()


def get_chirp_ifreq(content):
    curr_conj_mul = []
    for idx in range(1, len(content)):
        curr_conj_mul.append(content[idx]*np.conj(content[idx-1]))
    curr_ifreq = [phase(conj_mul) for conj_mul in curr_conj_mul]
    curr_ifreq.append(curr_ifreq[-1])
    return curr_ifreq


def show_packet(begin_seq):
    global packets_mlora2
    global d_samples_per_symbol
    global pre_peak
    global detect_scope

    idx = (begin_seq - 6) * d_samples_per_symbol
    while idx < len(packets_mlora2):
        if idx + d_samples_per_symbol > len(packets_mlora2):
            break

        bgn_idx = idx
        end_idx = idx + d_samples_per_symbol

        chirp_o = packets_mlora2[bgn_idx:end_idx]
        chirp_ifreq = get_chirp_ifreq(chirp_o)
        chirp_index, chirp_max, chirp_bin = get_fft_bins(chirp_o, d_samples_per_symbol, detect_scope, pre_peak)

        chirp_seq = idx / d_samples_per_symbol
        if chirp_seq < begin_seq+44:
            show_signal(chirp_ifreq, 'ifreq'+str(chirp_seq))
            show_signal(chirp_bin, 'fft'+str(chirp_seq))
            idx += d_samples_per_symbol
        else:
            break


def build_ideal_chirp():
    global d_samples_per_symbol
    global d_symbols_per_second
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


def get_fft_bins(content, window, scope, num):
    global d_downchirp
    global d_number_of_bins
    global d_samples_per_symbol
    curr_mul = np.multiply(content, d_downchirp[:window])
    curr_fft = np.abs(fft(curr_mul))

    # # only considering the previous half positive frequency bins and the last half negative frequency bins
    # # in total (0~d_decim_factor*d_bw)
    # curr_decim_fft = np.concatenate((curr_fft[:d_half_number_of_bins],
    #                                  curr_fft[(d_samples_per_symbol-d_half_number_of_bins):d_samples_per_symbol]))
    #
    # # positive frequency bins
    # curr_decim_fft = curr_fft[:d_number_of_bins]
    #
    # # negative frequency bins
    # curr_decim_fft = curr_fft[d_samples_per_symbol - d_number_of_bins:]

    # sum of pos and neg frequency bins
    curr_decim_fft = np.add(curr_fft[:d_number_of_bins], curr_fft[d_samples_per_symbol - d_number_of_bins:])

    # curr_decim_fft = curr_fft

    curr_fft = curr_decim_fft
    # curr_max = np.amax(curr_fft)
    # curr_index = np.where((curr_max - curr_fft) < 0.1)[0]
    # curr_index = int(np.sum(curr_index) / len(curr_index))
    curr_index, curr_maxbin = get_max_bins(curr_fft, scope, num)

    return curr_index, curr_maxbin, curr_fft


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


def get_down_chirp_bin(content, scope, num):
    global d_upchirp_zero
    global d_samples_per_symbol
    global d_half_number_of_bins

    curr_mul = np.multiply(content, d_upchirp_zero)
    curr_fft = np.abs(fft(curr_mul))

    # only considering the previous half positive frequency bins and the last half negative frequency bins
    # in total (0~d_decim_factor*d_bw)
    curr_decim_fft = np.concatenate((curr_fft[:d_half_number_of_bins],
                                     curr_fft[(d_samples_per_symbol - d_half_number_of_bins):d_samples_per_symbol]))

    curr_fft = curr_decim_fft
    curr_index, curr_maxbin = get_max_bins(curr_fft, scope, num)

    return curr_index, curr_maxbin, curr_fft


def detect_preamble_all(pre_stacks, prebin_stacks):
    global pre_chirps
    global pre_tol
    global bin_thr

    preamble_indexes = np.array([-idx - 1 for idx in range(len(pre_stacks))])
    preamble_bins = np.full(len(pre_stacks), -1, dtype=float)
    for idx in range(len(pre_stacks)):
        pre_idx = pre_stacks[idx][0]
        pre_found = True

        curr_idx = []
        curr_pos = []
        for jdx in range(pre_chirps):
            curr_found = False
            for kdx in range(len(pre_stacks)):
                if abs(pre_idx - pre_stacks[kdx][jdx]) < pre_tol and \
                        prebin_stacks[kdx][jdx] > bin_thr:
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
            preamble_indexes[idx] = pre_idx
            # save max bin of preamble, the first chirp is excluded to avoid incompleteness
            preamble_bins[idx] = np.average([prebin_stacks[bin_idx[1]][bin_idx[0]] for bin_idx in curr_idx[1:]])

    return preamble_indexes, preamble_bins


def detect_preamble_chirp(pkt_idx, bgn_idx, scope, num):
    global packets_mlora2
    global d_samples_per_symbol
    global pre_idxes
    global pre_tol
    global bin_thr

    chirp_o = packets_mlora2[bgn_idx:bgn_idx + d_samples_per_symbol]
    chirp_index, chirp_max, chirp_bin = get_fft_bins(chirp_o, d_samples_per_symbol, scope, num)

    is_preamble_chirp = False

    for idx in range(len(chirp_index)):
        if chirp_max[idx] > bin_thr and abs(chirp_index[idx] - pre_idxes[pkt_idx]) < pre_tol:
            is_preamble_chirp = True
            break

    return is_preamble_chirp


def detect_down_chirp(bgn_idx, scope, num):
    global packets_mlora2
    global d_samples_per_symbol
    global d_half_samples_per_symbol
    global sfd_history
    global sfdbin_history
    global sfd_chirps
    global sfd_thr
    global sfd_tol
    global sfd_idxes
    global sfd_bins

    sfd_history.clear()
    sfdbin_history.clear()
    sfdbins_history.clear()

    for idx in range(sfd_chirps):
        pad_chirp = np.zeros(d_samples_per_symbol, dtype=np.complex)
        curr_chirp = packets_mlora2[bgn_idx + idx * d_samples_per_symbol:bgn_idx + (idx + 1) * d_samples_per_symbol]
        pad_chirp[0:len(curr_chirp)] = curr_chirp
        sfd_idx, sfd_max, sfd_bin = get_down_chirp_bin(pad_chirp, scope, num)
        sfd_history.append(sfd_idx)
        sfdbin_history.append(sfd_max)
        sfdbins_history.append(sfd_bin)

        # sfd_ifreq = get_chirp_ifreq(pad_chirp)
        # show_signal(sfd_bin, 'fft')
        # show_signal(sfd_ifreq, 'ifreq')

    sfd_found = True

    print('sfd_history   : ', sfd_history)
    print('sfdbin_history: ', sfdbin_history)

    curr_idx = []
    for idx in range(sfd_chirps):
        curr_found = False
        for jdx in range(num):
            if abs(sfd_history[idx][jdx] - d_half_number_of_bins) <= sfd_tol \
                    and sfdbin_history[idx][jdx] > sfd_thr:
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
        sfd_idxes.append([sfd_history[bin_idx[0]][bin_idx[1]] for bin_idx in curr_idx])
        sfd_bins.append(np.sum([sfdbin_history[bin_idx[0]][bin_idx[1]] for bin_idx in curr_idx]))

        return True
    else:
        return False


def recover_single_packet():
    global pld_idxes
    global pld_bins
    global pld_chirps
    global rec_idxes
    global rec_bins

    for idx in range(len(pld_idxes)):
        if len(pld_idxes[idx]) == pld_chirps:
            for jdx in range(pld_chirps):
                rec_idxes[idx].append(pld_idxes[idx][jdx][np.argmax(pld_bins[idx][jdx])])
                rec_bins[idx].append(np.max(pld_bins[idx][jdx]))


def recover_packets():
    global d_number_of_bins
    global d_quad_number_of_bins
    global pre_idxes
    global detect_offset
    global detect_samp_offset
    global detect_chirp_offset
    global detect_bin_offset

    global pld_idxes
    global pld_bins
    global rec_idxes
    global rec_bins
    
    pre_idx = (pre_idxes[1] - pre_idxes[0] + d_quad_number_of_bins + d_number_of_bins) % d_number_of_bins

    # payload of A is collision-free
    for idx in range(0, max(detect_chirp_offset - 12, 0)):
        rec_idxes[0].append(pld_idxes[0][idx][np.argmax(pld_bins[0][idx])])
        rec_bins[0].append(np.max(pld_bins[0][idx]))
    
    # payload of A is overlapped by preamble or SFD of B
    for idx in range(max(detect_chirp_offset - 12, 0), min(pld_chirps, detect_chirp_offset)):
        curr_idx = []
        curr_max = []

        # excluding preamble chirp index of B
        for jdx in range(len(pld_idxes[0][idx])):
            if abs(pld_idxes[0][idx][jdx] - pre_idx) >= pld_tor \
                    and pld_bins[0][idx][jdx] > bin_thr:
                curr_idx.append(pld_idxes[0][idx][jdx])
                curr_max.append(pld_bins[0][idx][jdx])

        if len(curr_idx) >= 1:
            rec_idxes[0].append(curr_idx[np.argmax(curr_max)])
            rec_bins[0].append(np.max(curr_max))
        else:
            rec_idxes[0].append(pld_idxes[0][idx][np.argmax(pld_bins[0][idx])])
            rec_bins[0].append(np.max(pld_bins[0][idx]))

    samp_offset_a = {1: detect_samp_offset, 2: d_samples_per_symbol - detect_samp_offset}
    samp_offset_a = np.array(sorted(samp_offset_a.items(), key=lambda x: x[1]))
    ref_seg_a = samp_offset_a[-1][0]

    # chirp of A is partly overlapped by the first payload chirp of B
    if detect_chirp_offset < pld_chirps:
        if pre_idxes[0] <= pre_idxes[1]:
            rec_idxes[0].append(pld_idxes[0][detect_chirp_offset])
            rec_bins[0].append(pld_bins[0][detect_chirp_offset])
        else:
            rec_idxes[0].append(pld_idxes[0][detect_chirp_offset][ref_seg_a])
            rec_bins[0].append(pld_bins[0][detect_chirp_offset][ref_seg_a])

    # payload chirp of A is overlapped by two payload chirps of B
    for idx in range(min(pld_chirps, detect_chirp_offset + 1), pld_chirps):
        # using the longest segment of A as reference
        curr_idx = pld_idxes[0][idx][ref_seg_a]
        curr_max = pld_bins[0][idx][ref_seg_a]

        # counting the number of repeated index
        curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
        for jdx in range(len(pld_idxes[0][idx])):
            for kdx in range(len(curr_idx)):
                for ldx in range(len(curr_idx)):
                    if abs(curr_idx[ldx] - pld_idxes[0][idx][jdx][kdx]) <= pld_tor and \
                            pld_bins[0][idx][jdx][kdx] > 0:
                        curr_cnt[ldx] += 1
                        # curr_idx[ldx].append(pld_indexes[0][idx][jdx][kdx])

        rpt_index = np.where(curr_cnt >= 3)[0]

        if len(rpt_index) == 1:
            # only one repeated index
            rec_idxes[0].append(curr_idx[rpt_index[0]])
            rec_bins[0].append(curr_max[rpt_index[0]])
        else:
            # save multiple repeated indexes
            rec_idxes[0].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
            rec_bins[0].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

    samp_offset_b = {1: d_samples_per_symbol - detect_samp_offset, 2: detect_samp_offset}
    samp_offset_b = sorted(samp_offset_b.items(), key=lambda x: x[1])
    ref_seg_b = samp_offset_b[-1][0]
    
    # payload chirp of B is overlapped by two payload chirps of A
    for idx in range(0, max(0, pld_chirps - detect_chirp_offset - 1)):
        # using the longest segment of B as reference
        curr_idx = pld_idxes[1][idx][ref_seg_b]
        curr_max = pld_bins[1][idx][ref_seg_b]
        
        # counting the number of repeated index
        curr_cnt = np.zeros(len(curr_idx), dtype=np.int8)
        for jdx in range(len(pld_idxes[1][idx])):
            for kdx in range(len(curr_idx)):
                for ldx in range(len(curr_idx)):
                    if abs(curr_idx[ldx] - pld_idxes[1][idx][jdx][kdx]) <= pld_tor:
                        curr_cnt[ldx] += 1
                        # curr_idx[ldx].append(pld_indexes[1][idx][jdx][kdx])

        rpt_index = np.where(curr_cnt >= 3)[0]

        if len(rpt_index) == 1:
            # only one repeated index
            rec_idxes[1].append(curr_idx[rpt_index[0]])
            rec_bins[1].append(curr_max[rpt_index[0]])
        else:
            # save multiple repeated indexes
            rec_idxes[1].append(np.array(curr_idx)[np.where(curr_cnt >= 2)[0]])
            rec_bins[1].append(np.array(curr_max)[np.where(curr_cnt >= 2)[0]])

    # chirp of B is partly overlapped by the last payload chirp of A
    if detect_chirp_offset < pld_chirps:
        if pre_idxes[0] <= pre_idxes[1]:
            rec_idxes[1].append(pld_idxes[1][pld_chirps - detect_chirp_offset - 1])
            rec_bins[1].append(pld_bins[1][pld_chirps - detect_chirp_offset - 1])
        else:
            rec_idxes[1].append(pld_idxes[1][pld_chirps - detect_chirp_offset - 1][ref_seg_b])
            rec_bins[1].append(pld_bins[1][pld_chirps - detect_chirp_offset - 1][ref_seg_b])

    # payload of B is collision free
    for idx in range(max(pld_chirps - detect_chirp_offset, 0), pld_chirps):
        rec_idxes[1].append(pld_idxes[1][idx][np.argmax(pld_bins[1][idx])])
        rec_bins[1].append(np.max(pld_bins[1][idx]))
        
    # cross decoding followed by power mapping
    # symbol recovery based on repeated index is failed for A
    for idx in range(detect_chirp_offset, pld_chirps):
        if not (type(rec_idxes[0][idx]) is int):
            curr_idx = []
            curr_bin = []
            oly_idx = []

            oly_pld_b = idx - detect_chirp_offset - 2 + ref_seg_a
            # payload chirp index of B when aligning with chirp of A
            if 0 <= oly_pld_b < pld_chirps:
                if type(rec_idxes[1][oly_pld_b]) is int:
                    oly_idx.append((rec_idxes[1][oly_pld_b] - detect_bin_offset) % d_number_of_bins)
            else:
                rec_idxes[0][idx] = pld_idxes[0][idx][np.argmax(pld_bins[0][idx])]
                rec_bins[0][idx] = np.max(pld_bins[0][idx])

                continue

            # excluding payload chirp index of B
            for jdx in range(len(rec_idxes[0][idx])):
                is_oly = False
                for kdx in range(len(oly_idx)):
                    if abs(rec_idxes[0][idx][jdx] - oly_idx[kdx]) < pld_tor:
                        is_oly = True
                        break
                if not is_oly:
                    curr_idx.append(rec_idxes[0][idx][jdx])
                    curr_bin.append(rec_bins[0][idx][jdx])

            if len(curr_idx) == 1:
                # cross decoding successes
                rec_idxes[0][idx] = curr_idx[0]
                rec_bins[0][idx] = curr_bin[0]
            else:
                # cross decoding is also failed
                # performing power mapping
                if sfd_bins[0] > sfd_bins[1]:
                    jdx = 0
                else:
                    jdx = 1

                if idx == detect_chirp_offset:
                    rec_idxes[0][idx] = pld_idxes[0][idx][jdx]
                    rec_bins[0][idx] = pld_bins[0][idx][jdx]
                else:
                    rec_idxes[0][idx] = pld_idxes[0][idx][ref_seg_a][jdx]
                    rec_bins[0][idx] = pld_bins[0][idx][ref_seg_a][jdx]

    # symbol recovery based on repeated index is failed for B
    for idx in range(0, pld_chirps - detect_chirp_offset):
        if not (type(rec_idxes[1][idx]) is int):
            curr_idx = []
            curr_bin = []
            oly_idx = []

            oly_pld_a = idx + detect_chirp_offset - 1 + ref_seg_b
            # payload chirp index of B when aligning with chirp of A
            if detect_chirp_offset <= oly_pld_a < pld_chirps:
                if type(rec_idxes[0][oly_pld_a]) is int:
                    oly_idx.append((rec_idxes[0][oly_pld_a] + detect_bin_offset) % d_number_of_bins)
            else:
                rec_idxes[1][idx] = pld_idxes[1][idx][np.argmax(pld_bins[1][idx])]
                rec_bins[1][idx] = np.max(pld_bins[1][idx])

                continue

            # excluding payload chirp index of A
            for jdx in range(len(rec_idxes[1][idx])):
                is_oly = False
                for kdx in range(len(oly_idx)):
                    if abs(rec_idxes[1][idx][jdx] - oly_idx[kdx]) < pld_tor:
                        is_oly = True
                        break
                if not is_oly:
                    curr_idx.append(rec_idxes[1][idx][jdx])
                    curr_bin.append(rec_idxes[1][idx][jdx])

            if len(curr_idx) == 1:
                # cross decoding successes
                rec_idxes[1][idx] = curr_idx[0]
                rec_bins[1][idx] = curr_bin[0]
            else:
                # cross decoding is also failed
                # performing power mapping
                if sfd_bins[0] > sfd_bins[1]:
                    jdx = 1
                else:
                    jdx = 0

                if idx == pld_chirps - detect_chirp_offset - 1:
                    rec_idxes[1][idx] = pld_idxes[1][idx][jdx]
                    rec_bins[1][idx] = pld_bins[1][idx][jdx]
                else:
                    rec_idxes[1][idx] = pld_idxes[1][idx][ref_seg_b][jdx]
                    rec_bins[1][idx] = pld_bins[1][idx][ref_seg_b][jdx]


def error_correct():
    global corr_idxes
    global rec_idxes
    global pld_chirps
    global d_number_of_bins

    global packet_header
    global packet_index

    for idx in range(len(rec_idxes)):
        if len(rec_idxes[idx]) == pld_chirps:
            corr_idx = []
            idx_error = 0

            for jdx in range(8):
                ogn_idx = rec_idxes[idx][jdx]
                curr_idx = np.int16(np.rint(ogn_idx/4.0))

                idx_error += packet_header[packet_index][jdx] - ogn_idx
                corr_idx.append(4*curr_idx)

            idx_error = np.rint(idx_error/8.0)

            if abs(idx_error) >= 2:
                idx_error = 0

            for jdx in range(8, pld_chirps):
                ogn_idx = rec_idxes[idx][jdx]
                curr_idx = np.int16((ogn_idx + idx_error) % d_number_of_bins)
                corr_idx.append(curr_idx)

            corr_idxes.append(corr_idx)


def get_chirp_error():
    global corr_idxes
    global pld_chirps
    global packet_chirp
    global packet_index
    global chirp_errors

    for idx in range(len(corr_idxes)):
        errors_list = np.abs(np.subtract(packet_chirp[packet_index], np.array([corr_idxes[idx]] * 2)))
        errors = [np.count_nonzero(errors_list[0]), np.count_nonzero(errors_list[1])]

        chirp_errors.append(np.min(errors))


def recover_byte():
    global corr_idxes
    global rec_bytes
    global lora_decoder

    for idx in range(len(corr_idxes)):
        rec_bytes.append(lora_decoder.decode(corr_idxes[idx].copy()))


def get_byte_error():
    global rec_bytes
    global packet_byte
    global packet_index

    global header_errors
    global byte_errors

    for idx in range(len(rec_bytes)):
        comp_result_0 = []
        comp_result_1 = []
        header_result = []

        for jdx in range(3):
            header_result.append(bin(packet_byte[packet_index][idx][jdx] ^ rec_bytes[idx][jdx]).count('1'))

        for jdx in range(0, len(rec_bytes[idx])):
            comp_result_0.append(bin(packet_byte[packet_index][0][jdx] ^ rec_bytes[idx][jdx]).count('1'))
            comp_result_1.append(bin(packet_byte[packet_index][1][jdx] ^ rec_bytes[idx][jdx]).count('1'))

        header_errors.append(np.sum(header_result))
        byte_errors.append(np.min([np.sum(comp_result_0), np.sum(comp_result_1)]))


def init_decoding():
    global pre_idxes
    global pre_bins
    global pre_bgns
    global pre_history
    global prebin_history

    global sfd_idxes
    global sfd_bins
    global sfd_bgns
    global sfd_history
    global sfdbin_history
    global sfdbins_history

    global shift_samps
    global pld_chirps
    global pld_peak
    global pld_tor
    global pld_idxes
    global pld_bins
    global rec_idxes
    global rec_bins
    global oly_idxes
    global corr_idxes
    global rec_bytes
    global chirp_errors
    global header_errors
    global byte_errors

    global d_states

    pre_idxes = []
    pre_bgns = []
    pre_bins = []
    pre_history = [[], [], []]
    prebin_history = [[], [], []]

    sfd_idxes = []
    sfd_bins = []
    sfd_bgns = []
    sfd_history = []
    sfdbin_history = []
    sfdbins_history = []

    shift_samps = []

    pld_idxes = [[], []]
    pld_bins = [[], []]
    rec_idxes = [[], []]
    rec_bins = [[], []]
    oly_idxes = []

    # result
    corr_idxes = []
    rec_bytes = []
    chirp_errors = []
    header_errors = []
    byte_errors = []

    # d_states = [State.S_RESET, State.S_RESET]
    d_states = [State.S_PREFILL, State.S_PREFILL]


def show_result():
    global corr_idxes
    global rec_bytes
    global chirp_errors
    global header_errors
    global byte_errors

    print('Decoding Result:')
    print('Correct   Index:', corr_idxes)
    print('Recover   Bytes:', rec_bytes)
    print('Chirp    Errors:', chirp_errors)
    print('Header   Errors:', header_errors)
    print('Byte     Errors:', byte_errors)


def save_result():
    global pre_idxes
    global pre_bins
    global pre_bgns
    global sfd_idxes
    global sfd_bins
    global sfd_bgns
    global corr_idxes
    global rec_bytes
    global chirp_errors
    global header_errors
    global byte_errors

    global corr_idxes_list
    global rec_bytes_list
    global chirp_errors_list
    global header_errors_list
    global byte_errors_list

    pre_idxes_list.append(pre_idxes)
    pre_bins_list.append(pre_bins)
    pre_bgns_list.append(pre_bgns)

    sfd_idxes_list.append(sfd_idxes)
    sfd_bins_list.append(sfd_bins)
    sfd_bgns_list.append(sfd_bgns)

    corr_idxes_list.append(corr_idxes.copy())
    rec_bytes_list.append(rec_bytes.copy())
    chirp_errors_list.append(chirp_errors)
    header_errors_list.append(header_errors)
    byte_errors_list.append(byte_errors)


def detect_packets():
    global packets_mlora2
    global d_samples_per_symbol
    global d_quad_samples_per_symbol
    global d_decim_factor
    global d_states
    global detect_scope

    global pre_peak
    global pre_history
    global prebin_history
    global pre_chirps
    global pre_idxes
    global pre_bins
    global pre_bgns
    global pre_idxes_list
    global pre_bins_list
    global pre_bgns_list

    global sfd_peak
    global sfd_idxes
    global sfd_bins
    global sfd_bgns
    global sfd_idxes_list
    global sfd_bins_list
    global sfd_bgns_list

    global shift_samps
    global pld_chirps
    global pld_peak
    global pld_tor
    global pld_idxes
    global pld_bins

    global detect_offset
    global detect_chirp_offset
    global detect_samp_offset
    global detect_bin_offset

    bgn_seq = 0
    end_seq = bgn_seq + 44
    idx = bgn_seq * d_samples_per_symbol

    idx = 0
    stop_cnt = 0
    while idx < len(packets_mlora2):
        if idx + d_samples_per_symbol > len(packets_mlora2):
            break

        bgn_idx = idx
        end_idx = idx + d_samples_per_symbol

        chirp_o = packets_mlora2[bgn_idx:end_idx]
        chirp_ifreq = get_chirp_ifreq(chirp_o)
        chirp_index, chirp_max, chirp_bin = get_fft_bins(chirp_o, d_samples_per_symbol, detect_scope, pre_peak)

        chirp_seq = idx // d_samples_per_symbol
        pre_bgn_fst = bgn_seq
        # if pre_bgn_fst < chirp_seq < pre_bgn_fst+44:
        #     show_signal(chirp_ifreq, 'ifreq'+str(chirp_seq))
        #     show_signal(chirp_bin, 'fft'+str(chirp_seq))

        for j in range(pre_peak):
            pre_history[j].append(chirp_index[j])
            prebin_history[j].append(chirp_max[j])

            if len(pre_history[j]) > pre_chirps:
                pre_history[j].pop(0)
                prebin_history[j].pop(0)

        if d_states[0] == State.S_PREFILL and d_states[1] == State.S_PREFILL:
            if len(pre_history[0]) >= pre_chirps:
                for j in range(2):
                    d_states[j] = State.S_DETECT_PREAMBLE
            else:
                idx += d_samples_per_symbol
                continue

        if d_states[0] == State.S_DETECT_PREAMBLE or d_states[1] == State.S_DETECT_PREAMBLE:
            detect_idxes, detect_bins = detect_preamble_all(pre_history, prebin_history)

            if d_states[0] == State.S_DETECT_PREAMBLE and d_states[1] == State.S_DETECT_PREAMBLE:
                if len(np.where(detect_idxes >= 0)[0]) > 0:
                    repeat_idx = np.where(detect_idxes >= 0)[0]
                    if len(repeat_idx) >= 2:
                        for j in range(2):
                            d_states[j] = State.S_SFD_SYNC
                            pre_idxes.append(detect_idxes[repeat_idx[j]])
                            pre_bins.append(detect_bins[repeat_idx[j]])
                            pre_bgns.append(chirp_seq)

                        # align with the first packet
                        # i -= preamble_indexes[0] * d_decim_factor
                        # preamble_indexes[1] += preamble_indexes[0]
                    else:
                        d_states[0] = State.S_SFD_SYNC
                        pre_idxes.append(detect_idxes[repeat_idx[0]])
                        pre_bins.append(detect_bins[repeat_idx[0]])
                        pre_bgns.append(chirp_seq)
                        # align with the first packet
                        # i -= preamble_indexes[0] * d_decim_factor

            if d_states[0] != State.S_DETECT_PREAMBLE and d_states[1] == State.S_DETECT_PREAMBLE:
                unique_idx = np.where((detect_idxes >= 0) & (detect_idxes != pre_idxes[0]))[0]
                repeat_idx = np.where(detect_idxes >= 0)[0]
                if len(unique_idx) > 0 and d_states[0] == State.S_SFD_SYNC and abs(
                        detect_idxes[unique_idx[0]] - pre_idxes[0]) >= pre_tol:
                    d_states[1] = State.S_SFD_SYNC
                    pre_idxes.append(detect_idxes[unique_idx[0]])
                    pre_bins.append(detect_bins[unique_idx[0]])
                    pre_bgns.append(chirp_seq)

                if len(repeat_idx) > 0 and d_states[0] != State.S_SFD_SYNC:
                    d_states[1] = State.S_SFD_SYNC
                    pre_idxes.append(detect_idxes[repeat_idx[0]])
                    pre_bins.append(detect_bins[repeat_idx[0]])
                    pre_bgns.append(chirp_seq)

        # Test preamble detection
        # if d_states[0] == State.S_SFD_SYNC and d_states[1] == State.S_SFD_SYNC:
        #     print('Preamble Success')
        #     print('Preamble Indexes: ', pre_idxes)
        #     print('Preamble Bins   : ', pre_bins)
        #     print('Preamble Begins : ', pre_bgns)
        #
        #     pre_idxes_list.append(pre_idxes)
        #     pre_bins_list.append(pre_bins)
        #     pre_bgns_list.append(pre_bgns)
        #
        #     pre_idxes = []
        #     pre_bins = []
        #     pre_bgns = []
        #     for j in range(pre_peak):
        #         pre_history[j].clear()
        #         prebin_history[j].clear()
        #
        #     d_states[0] = State.S_PREFILL
        #     d_states[1] = State.S_PREFILL
        #     idx += d_samples_per_symbol
        #     continue
        #
        # else:
        #     idx += d_samples_per_symbol
        #     continue

        if d_states[0] == State.S_SFD_SYNC or d_states[1] == State.S_SFD_SYNC:

            if d_states[0] == State.S_SFD_SYNC and d_states[1] != State.S_SFD_SYNC:
                bgn_sfd = idx - pre_idxes[0] * d_decim_factor
                if not detect_preamble_chirp(0, idx, detect_scope, sfd_peak) \
                        and detect_down_chirp(bgn_sfd, detect_scope, sfd_peak):
                    d_states[0] = State.S_READ_PAYLOAD
                    sfd_bgns.append(bgn_sfd)
                    shift_samps.append(2 * d_samples_per_symbol + d_quad_samples_per_symbol - pre_idxes[0] * d_decim_factor)

            if d_states[0] != State.S_SFD_SYNC and d_states[1] == State.S_SFD_SYNC:
                bgn_sfd = idx - pre_idxes[1] * d_decim_factor
                if not detect_preamble_chirp(1, idx, detect_scope, sfd_peak) \
                        and detect_down_chirp(bgn_sfd, detect_scope, sfd_peak):
                    d_states[1] = State.S_READ_PAYLOAD
                    sfd_bgns.append(bgn_sfd)
                    shift_samps.append(2 * d_samples_per_symbol + d_quad_samples_per_symbol - pre_idxes[1] * d_decim_factor)

            if d_states[0] == State.S_SFD_SYNC and d_states[1] == State.S_SFD_SYNC:
                succ_cnt = 0

                bgn_sfd = idx - pre_idxes[0] * d_decim_factor
                if not detect_preamble_chirp(0, idx, detect_scope, sfd_peak) \
                        and detect_down_chirp(bgn_sfd, detect_scope, sfd_peak):
                    d_states[0] = State.S_READ_PAYLOAD
                    sfd_bgns.append(bgn_sfd)
                    shift_samps.append(2 * d_samples_per_symbol + d_quad_samples_per_symbol - pre_idxes[0] * d_decim_factor)
                    succ_cnt += 1

                bgn_sfd = idx - pre_idxes[1] * d_decim_factor
                if not detect_preamble_chirp(1, idx, detect_scope, sfd_peak) \
                        and detect_down_chirp(bgn_sfd, detect_scope, sfd_peak):
                    d_states[1] = State.S_READ_PAYLOAD
                    sfd_bgns.append(bgn_sfd)
                    shift_samps.append(2 * d_samples_per_symbol + d_quad_samples_per_symbol - pre_idxes[1] * d_decim_factor)
                    succ_cnt += 1

                # 同时检测到两个包的preamble和SFD, 保证数据包按序
                if succ_cnt == 2:
                    # reverse preamble index
                    if sfd_bgns[1] < sfd_bgns[0]:
                        sfd_bgns.reverse()
                        shift_samps.reverse()
                        pre_idxes.reverse()

                # 同时检测到两个包的preamble, 只检测到一个数据包的SFD
                if succ_cnt == 1:
                    if d_states[0] == State.S_SFD_SYNC and d_states[1] == State.S_READ_PAYLOAD:
                        d_states[0] = State.S_READ_PAYLOAD
                        d_states[1] = State.S_SFD_SYNC
                        pre_idxes.reverse()

        # Test packet synchronization
        # if d_states[0] == State.S_READ_PAYLOAD and d_states[1] == State.S_READ_PAYLOAD:
        #     print('Preamble Success')
        #     print('Preamble Indexes: ', pre_idxes)
        #     print('Preamble Bins   : ', pre_bins)
        #     print('Preamble Begins : ', pre_bgns)
        #
        #     pre_idxes_list.append(pre_idxes)
        #     pre_bins_list.append(pre_bins)
        #     pre_bgns_list.append(pre_bgns)
        #
        #     print('SFD      Success')
        #     print('SFD      Indexes: ', sfd_idxes)
        #     print('SFD      Bins   : ', sfd_bins)
        #     print('SFD      Begins : ', sfd_bgns)
        #
        #     sfd_idxes_list.append(sfd_idxes)
        #     sfd_bins_list.append(sfd_bins)
        #     sfd_cnts_list.append(sfd_bgns)
        #
        #     init_decoding()
        #
        #     idx += d_samples_per_symbol
        #     continue
        # else:
        #     idx += d_samples_per_symbol
        #     continue

        if d_states[0] == State.S_READ_PAYLOAD or d_states[1] == State.S_READ_PAYLOAD:

            if d_states[0] == State.S_READ_PAYLOAD and d_states[1] != State.S_READ_PAYLOAD:
                stop_cnt = 0

                if len(pld_idxes[0]) < pld_chirps:
                    chirp_0 = np.zeros(d_samples_per_symbol, dtype=np.complex)

                    bgn_index_0 = idx + shift_samps[0]
                    end_index_0 = bgn_index_0 + d_samples_per_symbol
                    chirp_0[0:len(packets_mlora2[bgn_index_0:end_index_0])] = packets_mlora2[bgn_index_0:end_index_0]

                    chirp_index_0, chirp_max_0, chirp_bin_0 = get_fft_bins(chirp_0, d_samples_per_symbol, detect_scope,
                                                                           pld_peak)

                    # chirp_index_0 = [chirp_index_0[j] for j in np.where(np.array(chirp_max_0) > bin_threshold)[0]]
                    # chirp_max_0 = [chirp_max_0[j] for j in np.where(np.array(chirp_max_0) > bin_threshold)[0]]

                    pld_idxes[0].append(chirp_index_0)
                    pld_bins[0].append(chirp_max_0)

                    # if d_debug == 1:
                    #     show_signal([chirp_bin_0, np.zeros(d_number_of_bins)],
                    #                 'Oly FFT 0 with index ' + str(len(pld_indexes[0]) - 1))
                    #     show_chirp(bgn_index_0, end_index_0, 0)

                if len(pld_idxes[0]) >= pld_chirps:
                    d_states[0] = State.S_STOP

            if d_states[0] != State.S_READ_PAYLOAD and d_states[1] == State.S_READ_PAYLOAD:

                if len(pld_idxes[1]) < pld_chirps:
                    # pre-assign to avoid the rest samples cannot form a complete chirp
                    chirp_1 = np.zeros(d_samples_per_symbol, dtype=np.complex)

                    bgn_index_1 = idx + shift_samps[1]
                    end_index_1 = bgn_index_1 + d_samples_per_symbol

                    chirp_1[0:len(packets_mlora2[bgn_index_1:end_index_1])] = packets_mlora2[bgn_index_1:end_index_1]

                    chirp_index_1, chirp_max_1, chirp_bin_1 = get_fft_bins(chirp_1, d_samples_per_symbol, detect_scope,
                                                                           pld_peak)

                    # chirp_index_1 = [chirp_index_1[j] for j in np.where(np.array(chirp_max_1) > threshold)[0]]
                    # chirp_max_1 = [chirp_max_1[j] for j in np.where(np.array(chirp_max_1) > threshold)[0]]

                    pld_idxes[1].append(chirp_index_1)
                    pld_bins[1].append(chirp_max_1)

                if len(pld_idxes[1]) >= pld_chirps:
                    d_states[1] = State.S_STOP

            if d_states[0] == State.S_READ_PAYLOAD and d_states[1] == State.S_READ_PAYLOAD:
                stop_cnt = 0

                detect_offset = sfd_bgns[1] - sfd_bgns[0]
                detect_chirp_offset = detect_offset // d_samples_per_symbol
                detect_samp_offset = detect_offset % d_samples_per_symbol
                detect_bin_offset = detect_samp_offset // d_decim_factor

                if len(pld_idxes[0]) < pld_chirps:
                    chirp_0 = np.zeros(d_samples_per_symbol, dtype=np.complex)
                    chirp_p0 = np.zeros(d_samples_per_symbol, dtype=np.complex)
                    chirp_b0 = np.zeros(d_samples_per_symbol, dtype=np.complex)

                    bgn_index_0 = idx + shift_samps[0]
                    end_index_0 = bgn_index_0 + d_samples_per_symbol
                    chirp_0[0:len(packets_mlora2[bgn_index_0:end_index_0])] = packets_mlora2[bgn_index_0:end_index_0]
                    chirp_p0[0:detect_samp_offset] = chirp_0[0:detect_samp_offset]
                    chirp_b0[detect_samp_offset:d_samples_per_symbol] = chirp_0[detect_samp_offset:d_samples_per_symbol]

                    chirp_index_0, chirp_max_0, chirp_bin_0 = get_fft_bins(chirp_0, d_samples_per_symbol, detect_scope,
                                                                           pld_peak)
                    chirp_index_p0, chirp_max_p0, chirp_bin_p0 = get_fft_bins(chirp_p0, d_samples_per_symbol,
                                                                              detect_scope, pld_peak)
                    chirp_index_b0, chirp_max_b0, chirp_bin_b0 = get_fft_bins(chirp_b0, d_samples_per_symbol,
                                                                              detect_scope, pld_peak)

                    # chirp_index_0 = [chirp_index_0[j] for j in np.where(np.array(chirp_max_0) > bin_threshold)[0]]
                    # chirp_max_0 = [chirp_max_0[j] for j in np.where(np.array(chirp_max_0) > bin_threshold)[0]]

                    pld_idxes[0].append([chirp_index_0, chirp_index_p0, chirp_index_b0])
                    pld_bins[0].append([chirp_max_0, chirp_max_p0, chirp_max_b0])

                    # if d_debug == 1:
                    #     show_signal([chirp_bin_0, np.zeros(d_number_of_bins)],
                    #                 'Oly FFT 0 with index ' + str(len(pld_indexes[0]) - 1))
                    #     show_chirp(bgn_index_0, end_index_0, 0)

                if len(pld_idxes[0]) >= pld_chirps:
                    d_states[0] = State.S_STOP

                if len(pld_idxes[1]) < pld_chirps:
                    # pre-assign to avoid the rest samples cannot form a complete chirp
                    chirp_1 = np.zeros(d_samples_per_symbol, dtype=np.complex)
                    chirp_p1 = np.zeros(d_samples_per_symbol, dtype=np.complex)
                    chirp_b1 = np.zeros(d_samples_per_symbol, dtype=np.complex)

                    bgn_index_1 = idx + shift_samps[1]
                    end_index_1 = bgn_index_1 + d_samples_per_symbol

                    chirp_1[0:len(packets_mlora2[bgn_index_1:end_index_1])] = packets_mlora2[bgn_index_1:end_index_1]
                    chirp_p1[0:d_samples_per_symbol - detect_samp_offset] \
                        = chirp_1[0:d_samples_per_symbol - detect_samp_offset]
                    chirp_b1[d_samples_per_symbol - detect_samp_offset:d_samples_per_symbol] = \
                        chirp_1[d_samples_per_symbol - detect_samp_offset:d_samples_per_symbol]

                    chirp_index_1, chirp_max_1, chirp_bin_1 = get_fft_bins(chirp_1, d_samples_per_symbol, detect_scope,
                                                                           pld_peak)
                    chirp_index_p1, chirp_max_p1, chirp_bin_p1 = get_fft_bins(chirp_p1, d_samples_per_symbol,
                                                                              detect_scope, pld_peak)
                    chirp_index_b1, chirp_max_b1, chirp_bin_b1 = get_fft_bins(chirp_b1, d_samples_per_symbol,
                                                                              detect_scope, pld_peak)

                    # chirp_index_1 = [chirp_index_1[j] for j in np.where(np.array(chirp_max_1) > threshold)[0]]
                    # chirp_max_1 = [chirp_max_1[j] for j in np.where(np.array(chirp_max_1) > threshold)[0]]

                    pld_idxes[1].append([chirp_index_1, chirp_index_p1, chirp_index_b1])
                    pld_bins[1].append([chirp_max_1, chirp_max_b1, chirp_max_p1])

                if len(pld_idxes[1]) >= pld_chirps:
                    d_states[1] = State.S_STOP

        if d_states[0] == State.S_STOP or d_states[1] == State.S_STOP:

            if d_states[0] == State.S_STOP and d_states[1] == State.S_DETECT_PREAMBLE:
                stop_cnt += 1

                if stop_cnt >= 6:
                    stop_cnt = 0
                    recover_single_packet()
                    error_correct()
                    get_chirp_error()
                    recover_byte()
                    get_byte_error()
                    show_result()
                    save_result()

                    init_decoding()

            if d_states[0] == State.S_STOP and d_states[1] == State.S_STOP:
                detect_offset = sfd_bgns[1] - sfd_bgns[0]
                detect_chirp_offset = detect_offset // d_samples_per_symbol
                detect_samp_offset = detect_offset % d_samples_per_symbol
                detect_bin_offset = detect_samp_offset // d_decim_factor

                recover_packets()
                error_correct()
                get_chirp_error()
                recover_byte()
                get_byte_error()
                show_result()
                save_result()

                init_decoding()

                # break

        idx += d_samples_per_symbol


def detect_single_packet():
    global packets_mlora2
    global d_samples_per_symbol
    global d_quad_samples_per_symbol
    global d_decim_factor
    global detect_scope

    global d_states
    global pre_peak
    global pre_history
    global prebin_history
    global pre_chirps
    global pre_idxes
    global pre_bins
    global pre_bgns
    global pre_idxes_list
    global pre_bins_list
    global pre_bgns_list

    global sfd_peak
    global sfd_idxes
    global sfd_bins
    global sfd_bgns
    global sfd_idxes_list
    global sfd_bins_list
    global sfd_bgns_list

    global shift_samps
    global pld_chirps
    global pld_peak
    global pld_tor
    global pld_idxes
    global pld_bins

    # idx = 1355 * d_samples_per_symbol
    idx = 0
    while idx < len(packets_mlora2):
        if idx + d_samples_per_symbol > len(packets_mlora2):
            break

        bgn_idx = idx
        end_idx = idx + d_samples_per_symbol

        chirp_o = packets_mlora2[bgn_idx:end_idx]
        chirp_ifreq = get_chirp_ifreq(chirp_o)
        chirp_index, chirp_max, chirp_bin = get_fft_bins(chirp_o, d_samples_per_symbol, detect_scope, pre_peak)

        chirp_seq = idx // d_samples_per_symbol
        # pre_bgn_fst = 671 - 6
        # if pre_bgn_fst < chirp_seq < pre_bgn_fst+44:
        #     show_signal(chirp_ifreq, 'ifreq'+str(chirp_seq))
        #     show_signal(chirp_bin, 'fft'+str(chirp_seq))

        for j in range(pre_peak):
            pre_history[j].append(chirp_index[j])
            prebin_history[j].append(chirp_max[j])

            if len(pre_history[j]) > pre_chirps:
                pre_history[j].pop(0)
                prebin_history[j].pop(0)

        if d_states[0] == State.S_PREFILL:
            if len(pre_history[0]) >= pre_chirps:
                d_states[0] = State.S_DETECT_PREAMBLE
            else:
                idx += d_samples_per_symbol
                continue

        if d_states[0] == State.S_DETECT_PREAMBLE:
            detect_idxes, detect_bins = detect_preamble_all(pre_history, prebin_history)

            if len(np.where(detect_idxes >= 0)[0]) > 0:
                repeat_idx = np.where(detect_idxes >= 0)[0]
                if len(repeat_idx) >= 2:
                    d_states[0] = State.S_SFD_SYNC
                    for j in range(2):
                        pre_idxes.append(detect_idxes[repeat_idx[j]])
                        pre_bins.append(detect_bins[repeat_idx[j]])
                        pre_bgns.append(chirp_seq)
                else:
                    d_states[0] = State.S_SFD_SYNC
                    pre_idxes.append(detect_idxes[repeat_idx[0]])
                    pre_bins.append(detect_bins[repeat_idx[0]])
                    pre_bgns.append(chirp_seq)

        # Test for preamble detection
        # if d_states[0] == State.S_SFD_SYNC:
        #     print('One Preamble Success')
        #     print('Preamble Indexes: ', pre_idxes)
        #     print('Preamble Bins   : ', pre_bins)
        #     print('Preamble Begins : ', pre_bgns)
        #
        #     pre_idxes_list.append(pre_idxes)
        #     pre_bins_list.append(pre_bins)
        #     pre_bgns_list.append(pre_bgns)
        #
        #     pre_idxes = []
        #     pre_bins = []
        #     pre_bgns = []
        #     for j in range(pre_peak):
        #         pre_history[j].clear()
        #         prebin_history[j].clear()
        #
        #     d_states[0] = State.S_PREFILL
        #     idx += d_samples_per_symbol
        #     continue
        #
        # else:
        #     idx += d_samples_per_symbol
        #     continue

        if d_states[0] == State.S_SFD_SYNC:
            bgn_sfd = idx - pre_idxes[0] * d_decim_factor
            if not detect_preamble_chirp(0, idx, detect_scope, sfd_peak) \
                    and detect_down_chirp(bgn_sfd, detect_scope, sfd_peak):
                d_states[0] = State.S_READ_PAYLOAD
                sfd_bgns.append(bgn_sfd)
                shift_samps.append(2*d_samples_per_symbol + d_quad_samples_per_symbol - pre_idxes[0] * d_decim_factor)

        # Test for packet synchronization
        # if d_states[0] == State.S_READ_PAYLOAD:
        #
        #     print('Preamble Success')
        #     print('Preamble Indexes: ', pre_idxes)
        #     print('Preamble Bins   : ', pre_bins)
        #     print('Preamble Begins : ', pre_bgns)
        #
        #     pre_idxes_list.append(pre_idxes)
        #     pre_bins_list.append(pre_bins)
        #     pre_bgns_list.append(pre_bgns)
        #
        #     print('SFD      Success')
        #     print('SFD      Indexes: ', sfd_idxes)
        #     print('SFD      Bins   : ', sfd_bins)
        #     print('SFD      Begins : ', sfd_bgns)
        #
        #     sfd_idxes_list.append(sfd_idxes)
        #     sfd_bins_list.append(sfd_bins)
        #     sfd_cnts_list.append(sfd_bgns)
        #
        #     init_decoding()
        #
        #     idx += d_samples_per_symbol
        #     continue
        # else:
        #     idx += d_samples_per_symbol

        if d_states[0] == State.S_READ_PAYLOAD:
            if len(pld_idxes[0]) < pld_chirps:
                curr_chirp = np.zeros(d_samples_per_symbol, dtype=np.complex)
                bgn_idx = idx + shift_samps[0]
                end_idx = bgn_idx + d_samples_per_symbol

                curr_chirp[0:len(packets_mlora2[bgn_idx:end_idx])] = packets_mlora2[bgn_idx:end_idx]

                chirp_index, chirp_max, chirp_bin = get_fft_bins(curr_chirp, d_samples_per_symbol, detect_scope, pld_peak)

                pld_idxes[0].append(chirp_index)
                pld_bins[0].append(chirp_max)

            if len(pld_idxes[0]) >= pld_chirps:
                d_states[0] = State.S_STOP

        if d_states[0] == State.S_STOP:
            print('Preamble Success')
            print('Preamble Indexes: ', pre_idxes)
            print('Preamble Bins   : ', pre_bins)
            print('Preamble Begins : ', pre_bgns)

            pre_idxes_list.append(pre_idxes)
            pre_bins_list.append(pre_bins)
            pre_bgns_list.append(pre_bgns)

            print('SFD      Success')
            print('SFD      Indexes: ', sfd_idxes)
            print('SFD      Bins   : ', sfd_bins)
            print('SFD      Begins : ', sfd_bgns)

            sfd_idxes_list.append(sfd_idxes)
            sfd_bins_list.append(sfd_bins)
            sfd_bgns_list.append(sfd_bgns)

            recover_single_packet()
            error_correct()
            get_chirp_error()
            recover_byte()
            get_byte_error()
            show_result()
            save_result()

            init_decoding()

            idx += d_samples_per_symbol
            continue
        else:
            idx += d_samples_per_symbol
            continue


path_prefix = ['../data/mlora2_sync', '../data/mlora2_async']

power = [2, 0]
byte_len = [8, 8]
chirp_len = [44, 44]
payload_len = [32, 32]
payload = ['fwd', 'inv']

packets_mlora2_sync = scipy.fromfile(path_prefix[0], scipy.complex64)
packets_mlora2_async = scipy.fromfile(path_prefix[1], scipy.complex64)

packets_mlora2 = packets_mlora2_async

packet_index = np.int(np.log2(byte_len[0])) - 3

packet_chirp = \
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

packet_header = np.array([[124, 0, 0, 76, 4, 108, 8, 120], [56, 0, 28, 124, 116, 96, 108, 40]])

packet_byte = np.array(
    [[[8, 128, 144, 18, 52, 86, 120, 18, 52, 86, 120, 0, 0, 0, 0, 0, 0, 0, 0],
      [8, 128, 144, 33, 67, 101, 135, 33, 67, 101, 135, 0, 0, 0, 0, 0, 0, 0, 0]],
     [[16, 129, 64, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120, 18, 52, 86, 120],
      [16, 129, 64, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135, 33, 67, 101, 135]]],
    dtype=np.uint8)

d_bw = 125e3
d_samples_per_second = 1e6
d_dt = 1.0/d_samples_per_second

d_sf = 7
d_cr = 4
d_decim_factor = int(d_samples_per_second/d_bw)
d_number_of_bins = (1 << d_sf)
d_symbols_per_second = d_bw/d_number_of_bins
d_samples_per_symbol = d_number_of_bins*d_decim_factor
d_half_number_of_bins = int(d_number_of_bins/2)
d_quad_number_of_bins = int(d_number_of_bins/4)
d_half_samples_per_symbol = int(d_samples_per_symbol/2)
d_quad_samples_per_symbol = int(d_samples_per_symbol/4)

d_downchirp = []
d_downchirp_zero = []
d_upchirp = []
d_upchirp_zero = []
build_ideal_chirp()

detect_scope = 2
bin_thr = 8

pre_peak = 3
pre_chirps = 6
pre_tol = 2
pre_idxes = []
pre_bgns = []
pre_bins = []
pre_history = [[], [], []]
prebin_history = [[], [], []]
pre_idxes_list = []
pre_bins_list = []
pre_bgns_list = []

sfd_peak = 3
sfd_thr = 8
sfd_chirps = 2
sfd_tol = 4
sfd_idxes = []
sfd_bins = []
sfd_bgns = []
sfd_history = []
sfdbin_history = []
sfdbins_history = []
sfd_idxes_list = []
sfd_bins_list = []
sfd_bgns_list = []

detect_offset = None
detect_chirp_offset = None
detect_samp_offset = None
detect_bin_offset = None
pld_peak = 2
pld_tor = 2
pld_chirps = int(8 + (4 + d_cr) * np.ceil(byte_len[0] * 2.0 / d_sf))
shift_samps = []
lora_decoder = LoRaDecode(byte_len[0], d_sf, d_cr)

pld_idxes = [[], []]
pld_bins = [[], []]
rec_idxes = [[], []]
rec_bins = [[], []]
oly_idxes = []

# result
corr_idxes = []
rec_bytes = []
chirp_errors = []
header_errors = []
byte_errors = []

# save result
corr_idxes_list = []
rec_bytes_list = []
chirp_errors_list = []
header_errors_list = []
byte_errors_list = []

# d_states = [State.S_RESET, State.S_RESET]
d_states = [State.S_PREFILL, State.S_PREFILL]

# MLoRa207171035
# pre_bgn_list = [672,  916, 1160, 1405, 1649, 1894, 2138, 2382, 2627, 2871]

# MLoRa207171425
# pre_bgns_list = [1012, 1256, 1501, 1745, 1989, 2233, 2477, 2722, 2966, 3210]

# MLoRa207171505
# pre_bgns_list = [900, 1145, 1389, 1633, 1877, 2121, 2365, 2609, 2854, 3098]
# pre_bgns_list = [[900,  974], [1144, 1389], [1462, 1633], [1706, 1877], [1951, 2121], [2195, 2365], [2439, 2609], [2683, 2853], [2927, 3097]]

# MLoRa207171545
# pre_bgns_list = [849, 1093, 1337, 1582, 1827, 2071, 2315, 2560, 2804, 3048]

# MLoRa207171550
# pre_bgn_list = [1124, 1369, 1613, 1857, 2102, 2346, 2591, 2835, 3084, 3329]
# pre_bgns_list = [[1051, 1124], [1296, 1368], [1540, 1613], [1785, 1857], [2030, 2102], [2274, 2346], [2518, 2591], [2763, 2835], [3012, 3084], [3256, 3329]]

# MLoRa207181900
# pre_bgn_list = [1044, 1117, 1288, 1361, 1532, 1606, 1777, 1850, 2021, 2095, 2265, 2339, 2509, 2583, 2754, 2827, 2998, 3071, 3242, 3316]

# Max FFT bins of node: [0, 0] TX power 0.5 cycle antenna
# max_bins = [{0: [10, 20]}, {1: [8, 10]}, {2: [15, 25]}]
# Node 0 SFD 63
# Node 1 SFD 52


init_decoding()
# detect_single_packet()
detect_packets()
# show_packet(974)
