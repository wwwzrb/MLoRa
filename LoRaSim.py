import matplotlib.pyplot as plt
import numpy as np
from cmath import phase
from scipy.fftpack import fft


class LoRaSim:

    def __init__(self, spreading_factor):
        self.sf = spreading_factor
        self.mod_len = (1 << self.sf)
        self.sfd = int(self.mod_len/4)

        self.std_up_chirp = []
        self.std_down_chirp = []
        self.gen_std_chirp()

    def gen_std_chirp(self):
        phase_shift = -np.pi
        accumulator = 0
        # self.std_down_chirp.append(-1+0j)
        # self.std_up_chirp.append(-1-0j)
        for i in range(2 * self.mod_len):
            accumulator += phase_shift
            self.std_down_chirp.append(np.conj(np.exp(1j * accumulator)))
            self.std_up_chirp.append(np.exp(1j * accumulator))
            phase_shift += (2 * np.pi) / self.mod_len

    def mod(self, chirps):
        res = []
        for i in range(len(chirps)):
            curr_chirp = []
            for j in range(self.mod_len):
                curr_chirp.append(self.std_up_chirp[(chirps[i] + j) % self.mod_len])
            res.append(curr_chirp)
        return res

    def de_mod(self, time_samples):
        res = []
        res_fft = []
        res_multi = []
        chirp_len = len(time_samples)
        for i in range(chirp_len):
            chirp = time_samples[i]
            downed_chirp = np.multiply(chirp, self.std_down_chirp[:self.mod_len])

            chirp_fft = np.abs(fft(downed_chirp))
            res_fft.append(chirp_fft)
            chirp_max = np.amax(chirp_fft)
            chirp_index = np.where((chirp_max - chirp_fft) < 0.1)[0]
            chirp_index = int(np.sum(chirp_index) / len(chirp_index))

            res_multi.append(downed_chirp)
            res.append(chirp_index % self.mod_len)
        return res, res_fft, res_multi

    def sync_detection(self, time_samples, overlap_factor):
        res = []
        res_fft = []

        for i in range(overlap_factor):
            offset = int(((i*self.mod_len)/overlap_factor) % self.mod_len)
            chirp = time_samples[0]
            downed_chirp = np.multiply(chirp, self.std_down_chirp[offset:offset+self.mod_len])

            chirp_fft = np.abs(fft(downed_chirp))
            res_fft.append(chirp_fft)
            chirp_max = np.amax(chirp_fft)
            chirp_index = np.where((chirp_max - chirp_fft) < 1e-3)[0]
            chirp_index = int(np.sum(chirp_index) / len(chirp_index))

            res.append(chirp_index % self.mod_len)
        return res, res_fft

    @staticmethod
    def show_freq(time_samples, index):
        curr_chirp = time_samples[index]
        phase_shift = []
        for i in range(1, len(curr_chirp)):
            phase_shift.append(curr_chirp[i] * np.conj(curr_chirp[i - 1]))
        phase_shift = [phase(conj_mul) for conj_mul in phase_shift]
        t = [i for i in range(len(phase_shift))]
        plt.plot(t, phase_shift)
        plt.show()
        return np.sum(phase_shift)

    @staticmethod
    def show_fft(freq_samples, index):
        curr_chirp = freq_samples[index]
        t = [i for i in range(len(curr_chirp))]
        plt.plot(t, curr_chirp)
        plt.show()
