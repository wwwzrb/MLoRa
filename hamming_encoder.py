import numpy as np


K = 4
#works only for 7,4


def encode(s):
    # Read in K=4 bits at a time and write out those plus parity bits
    while len(s) >= K:
        nibble = s[0:K]
        print(hamming(nibble))
        s = s[K:]


def hamming(bits):
    # Return given 4 bits plus parity bits for bits (1,2,3), (2,3,4) and (1,3,4)
    t1 = parity(bits, [0, 1, 3])
    t2 = parity(bits, [0, 2, 3])
    t3 = parity(bits, [1, 2, 3])
    ham74 = t1 + t2 + bits[0] + t3 + bits[1:]
    t4 = parity(ham74, [i for i in range(7)])
    ham84 = ham74 + t4
    return ham84  # again saying, works only for 7,4


def parity(s, indicies):
    # Compute the parity bit for the given string s and indicies
    sub = ""
    for i in indicies:
        sub += s[i]
    return str(str.count(sub, "1") % 2)


if __name__ == "__main__":
    print("Enter Input String of bits - ")  # just for testing
    input_string = input().strip()
    while input_string:
        print(" Output - ", end=' ')  # just for testing
        encode(input_string)

        input_string = input().strip()
