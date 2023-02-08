import numpy as np
import threading
import time


class SendThread(threading.Thread):

    def __init__(self, thread_id, device, repeats, content, intra_delay, intra_interval):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.device = device
        self.times = repeats
        self.payload = content
        self.intra_delay = intra_delay
        self.intra_interval = intra_interval

    def run(self):
        print("开始发送: " + str(self.thread_id) + str(time.time()))
        self.send_data()
        print("结束发送: " + str(self.thread_id) + str(time.time()))

    def send_data(self):
        for idx in range(self.times):
            time.sleep(self.intra_delay[idx])
            print("Thread {} : {} : {}".format(self.thread_id, idx, str(time.time())))
            time.sleep(self.intra_interval[idx])


# Multiple thread test
# Divide sending time of single packet 0.08s into sections
section_num = 20
least_delay = 0.020/section_num
section_cnts = []
intra_delays = [[], []]
intra_intervals = [[], []]

basic_seq_0 = "12345678"
basic_seq_1 = "21436587"
basic_seq_2 = "87654321"
basic_seq = [basic_seq_0, basic_seq_1, basic_seq_2]

seq_len = 8
seq_idx = [0, 1]
seq = ['', '']

for i in range(seq_len // 4):
    for j in range(len(seq_idx)):
        seq[j] += basic_seq[seq_idx[j]]

for i in range(len(seq_idx)):
    print('Payload {} : {}'.format(i, seq[i]))

times = 10
for i in range(times):
    intra_delays[0].append(0)
    section_cnt = np.random.randint(1, high=section_num + 1)
    section_cnts.append(section_cnt)
    intra_delays[1].append(least_delay * section_cnt)
    # intra_delays[1].append(0)

    intra_intervals[0].append(0.1)
    intra_intervals[1].append(0.1 - intra_delays[1][i])

send_thread0 = SendThread(0, 0, times, seq[0], intra_delays[0], intra_intervals[0])
send_thread1 = SendThread(1, 1, times, seq[1], intra_delays[1], intra_intervals[1])

send_thread0.start()
send_thread1.start()
send_thread0.join()
send_thread1.join()
print("退出主线程")
