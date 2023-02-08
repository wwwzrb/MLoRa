# -*- coding: UTF-8 -*-
from enum import Enum, unique


@unique
class State(Enum):
    S_RESET = 0
    S_PREFILL = 1
    S_DETECT_PREAMBLE = 2
    S_SFD_SYNC = 3
    S_SFD_CONSUME = 4
    S_READ_PAYLOAD = 5
    S_STOP = 6


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def pop_back(self):
        return self.items.pop(0)

    def peek(self):
        return self.items[len(self.items) - 1]

    def bottom(self):
        return self.items[0]

    def size(self):
        return len(self.items)

    def get_list(self):
        return self.items

    def get_i(self, i):
        return self.items[i]

    def set_i(self, i, value):
        self.items[i] = value

    def clear(self):
        self.items = []


# d_state = State.S_RESET
# if d_state is State.S_RESET:
#     print(d_state)
# s = Stack()
#
# print(s.is_empty())
# s.push(4)
# s.push('dog')
# print(s.peek())
# s.push(True)
# print(s.size())
# print(s.is_empty())
# s.push(8.4)
# print(s.pop_back())
# print(s.pop_back())
# print(s.size())
# print(s.get_list())
# print(s.bottom())
