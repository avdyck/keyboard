import math
import random
from typing import List

import numpy as np

"""
<   o   l   f   y   q   p   m   u   b
 i   a   n   s   k   v   t   h   e   r   j
   >   "   d   g   x   w   c   ?   ;   z
w   m   o   f   >   x   j   u   s   k
 d   n   a   i   b   g   r   e   h   t   v
   ;   ?   y   p   "   l   c   <   z   q
f   <   o   q   m   l   b   p   y   u
 i   n   a   s   k   v   t   h   e   r   j
   >   "   x   d   g   c   w   ?   ;   z
i   n   a   l   d   q   r   e   t   h
 y   f   o   p   v   x   s   u   b   m   k
   "   ?   >   c   g   w   j   <   ;   z
"""

keys = [
    # """
    # 'q',  'w',  'e',  'r',  't',  'y',  'u',  'i',  'o',  'p',
    #  'a',  's',  'd',  'f',  'g',  'h',  'j',  'k',  'l',  ';',  '"',
    #     'z',  'x',  'c',  'v',  'b',  'n',  'm',  '<',  '>',  '?'
    'p',  'l',  'o',  'k',  'q',  'c',  'w',  'u',  's',  'v',
     'i',  'n',  'a',  'd',  'f',  '.',  'r',  'e',  'h',  't',  'g',
       'x',  '.',  '.',  'm',  'z',  'j',  'b',  'y',  '.',  '.',
    # """
]

effort = np.array([
    3.5,  1.5,  1.5,  2.0,  2.5,  3.0,  2.0,  1.5,  1.5,  2.5,
     1.2,  1.0,  1.0,  1.0,  2.0,  2.0,  1.0,  1.0,  1.0,  1.2,   3.0,
        4.0,  4.0,  1.5,  1.5,  3.0,  1.5,  1.5,  2.5,  3.0,  3.0,
], dtype=float)

keymap = {k: i for i, k in enumerate(keys)}

rows = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
], dtype=int)

cols = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
], dtype=int)

a = 0
b = 1
c = 2
roll_map = np.array([
    [  # q
        a, c, b, b, b, b, b, b, b, b,
        0, 0, b, b, b, b, b, b, b, b, b,
        0, 0, b, b, b, b, b, b, b, b
    ], [  # w
        a, a, c, b, b, b, b, b, b, b,
        b, 0, b, b, b, b, b, b, b, b, b,
        0, 0, b, b, b, b, b, b, b, b
    ], [  # e
        a, a, a, c, b, b, b, b, b, b,
        b, b, 0, b, b, b, b, b, b, b, b,
        0, 0, 0, b, b, b, b, b, b, b
    ], [  # r
        a, a, a, a, 0, b, b, b, b, b,
        b, b, 0, 0, 0, b, b, b, b, b, b,
        0, 0, 0, 0, 0, b, b, b, b, b
    ], [  # t
        a, a, a, a, a, b, b, b, b, b,
        b, b, b, 0, 0, b, b, b, b, b, b,
        0, 0, 0, 0, 0, b, b, b, b, b
    ], [  # y
        a, a, a, a, a, a, 0, b, b, b,
        b, b, b, b, b, 0, 0, b, b, b, b,
        b, b, b, b, b, 0, 0, 0, 0, 0
    ], [  # u
        a, a, a, a, a, a, a, c, b, b,
        b, b, b, b, b, 0, 0, b, b, b, b,
        b, b, b, b, b, 0, 0, 0, 0, b
    ], [  # i
        a, a, a, a, a, a, a, a, c, b,
        b, b, b, b, b, b, b, 0, b, b, b,
        b, b, b, b, b, b, b, 0, b, b
    ], [  # o
        a, a, a, a, a, a, a, a, a, c,
        b, b, b, b, b, b, b, b, 0, b, 0,
        b, b, b, b, b, b, b, 0, 0, 0
    ], [  # p
        a, a, a, a, a, a, a, a, a, a,
        b, b, b, b, b, b, b, b, b, 0, 0,
        b, b, b, b, b, b, b, 0, 0, 0
    ], [  # a
        a, a, a, a, a, a, a, a, a, a,
        a, c, b, b, b, b, b, b, b, b, b,
        0, b, b, b, b, b, b, b, b, b
    ], [  # s
        a, a, a, a, a, a, a, a, a, a,
        a, a, c, b, b, b, b, b, b, b, b,
        0, 0, b, b, b, b, b, b, b, b
    ], [  # d
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, c, b, b, b, b, b, b, b,
        b, 0, 0, b, b, b, b, b, b, b
    ], [  # f
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, 0, b, b, b, b, b, b,
        b, 0, 0, 0, 0, b, b, b, b, b
    ], [  # g
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, b, b, b, b, b, b,
        b, 0, 0, 0, 0, b, b, b, b, b
    ], [  # h
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, 0, b, b, b, b,
        b, b, b, b, b, 0, 0, 0, 0, b
    ], [  # j
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, c, b, b, b,
        b, b, b, b, b, 0, 0, b, b, b
    ], [  # k
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, c, b, b,
        b, b, b, b, b, b, b, 0, b, 0
    ], [  # l
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, c, b,
        b, b, b, b, b, b, b, 0, 0, 0
    ], [  # ;
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, 0,
        b, b, b, b, b, b, b, b, b, 0
    ], [  # '
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        b, b, b, b, b, b, b, b, 0, 0
    ], [  # z
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, c, b, b, b, b, b, b, b, b
    ], [  # x
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, c, b, b, b, b, b, b, b
    ], [  # c
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, 0, 0, b, b, b, b, b
    ], [  # v
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, 0, b, b, b, b, b
    ], [  # b
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, b, b, b, b, b
    ], [  # n
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, 0, b, b, b
    ], [  # m
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, c, b, b
    ], [  # ,
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, c, b
    ], [  # .
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, c
    ], [  # /
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a
    ]
], dtype=int)

neighbors = np.array([
    [  # q
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # w
        1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # e
        0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # r
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # t
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # y
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # u
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # i
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # o
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # p
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # 0
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # s
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0
    ], [  # d
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0
    ], [  # f
        0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 0
    ], [  # g
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0
    ], [  # h
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0
    ], [  # j
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0
    ], [  # k
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0
    ], [  # l
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1
    ], [  # ;
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1
    ], [  # '
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    ], [  # z
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0
    ], [  # x
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 0, 0, 0, 0, 0
    ], [  # c
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0, 0, 0
    ], [  # v
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0
    ], [  # 0
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0
    ], [  # n
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0
    ], [  # m
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0
    ], [  # ,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0
    ], [  # .
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1
    ], [  # /
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    ]
], dtype=int)

letter_frequency: np.ndarray = np.zeros(len(keys), dtype=np.float)
digrams: np.ndarray = np.zeros((len(keys), len(keys)), dtype=float)


def initialize():

    def initialize_letters():
        with open('freq-letter.txt') as fh:
            for line in fh:
                l, f = line.strip().split('\t')
                letter_frequency[keymap[l]] = float(f)

    def initialize_digrams():
        with open('freq-digram.txt') as fh:
            for line in fh:
                (l0, l1), f = line.strip().split('\t')
                digrams[keymap[l0], keymap[l1]] = float(f)

    # def initialize_trigrams():
    #     trigrams = []
    #     with open('freq-trigram.txt') as fh:
    #         for line in fh:
    #             (l0, l1, l2), f = line.strip().split('\t')
    #             f = float(f)
    #             if f >= 0.0001:
    #                 trigrams.append(((keymap[l0], keymap[l1], keymap[l2]), f))
    #
    #     for key in range(len(keys)):
    #         sub_trigrams = []
    #         sub_frequencies = []
    #         for trigram, freq in trigrams:
    #             if key in trigram:
    #                 sub_trigrams.append(trigram)
    #                 sub_frequencies.append(freq)
    #
    #         trigrams_containing.append(np.array(sub_trigrams))
    #         trigram_frequency_containing.append(np.array(sub_frequencies))

    def initialize_roll_map():
        i = np.tril_indices_from(roll_map)
        roll_map[i] = roll_map.T[i]

    initialize_letters()
    initialize_digrams()
    # initialize_trigrams()
    initialize_roll_map()


class Keyboard:
    def __init__(self):
        self.keyboard = np.array(range(len(keys)), dtype=int)
        self.location = np.array(range(len(keys)), dtype=int)
        self.penalty = 0.
        self.calculate_penalties()

    def neighbor(self):
        """
        >>> kb = Keyboard()
        >>> for i in range(1000):
        ...     n0, n1 = kb.neighbor()
        ...     assert n0 != n1
        """
        key1letter = 'e'
        key2letter = 'a'
        i1 = -1
        i2 = -1
        while {key1letter, key2letter} & {'a', 'e', 'u', 'o'}:
            i1 = random.randint(1, len(self.keyboard) - 1)
            # key1 = self.keyboard[i1]
            # i2 = random.choice(np.where(neighbors[key1] == 1)[0])
            i2 = random.randint(1, len(self.keyboard) - 1)
            if i1 == i2:
                i2 = 0
            key1letter = keys[self.keyboard[i1]]
            key2letter = keys[self.keyboard[i2]]

        return i1, i2

    def calculate_penalties(self):
        # letter
        key = self.keyboard
        letter_delta = np.sum(letter_frequency[key] * effort)

        # digram
        digram_delta = 0
        for i0, k0 in enumerate(key):
            for i1, k1 in enumerate(key):
                digram_delta -= roll_map[i0, i1] * digrams[k0, k1]

        self.penalty = letter_delta + digram_delta

    def score_and_apply_neighbor(self, i1: int, i2: int):
        """
        >>> kb = Keyboard()
        >>> kb.score_and_apply_neighbor(keymap['x'], keymap['s'])
        >>> kb.penalty > Keyboard().penalty
        True
        """
        self.apply_neighbor(i1, i2)
        self.calculate_penalties()

    def apply_neighbor(self, i1: int, i2: int) -> None:
        key1 = self.keyboard[i1]
        key2 = self.keyboard[i2]
        assert self.location[key1] == i1
        assert self.location[key2] == i2
        self.location[key1] = i2
        self.location[key2] = i1
        self.keyboard[i1] = key2
        self.keyboard[i2] = key1

    def optimize(self, iterations: int):
        ii = jj = 0
        best = self.deepcopy()
        curr = self.deepcopy()
        for i in range(iterations):
            if i % (iterations // 100) == 0:
                print(end='.')
            t = math.exp((iterations - i - 1) / (iterations)) - 1
            prev = curr.deepcopy()
            i1, i2 = curr.neighbor()
            curr.score_and_apply_neighbor(i1, i2)
            if curr.penalty < prev.penalty + t:
                # We're good
                ii += 1
                if curr.penalty < best.penalty:
                    best = curr.deepcopy()
            else:
                # Undo
                jj += 1
                curr = prev

        print('\n', ii, jj, curr.penalty, best.penalty)
        return best

    def deepcopy(self):
        result = Keyboard()
        result.keyboard = self.keyboard.copy()
        result.location = self.location.copy()
        result.penalty = self.penalty
        return result

    def __repr__(self):
        """
        >>> Keyboard()
        p   l   o   k   q   c   w   u   s   v
         i   n   a   d   f   .   r   e   h   t   g
           x   .   .   m   z   j   b   y   .   .
        """
        return str(self)

    def __str__(self):
        result = []
        for k in self.keyboard[:10]:
            result.append(keys[k])
            result.append('   ')
        result.pop()
        result.append('\n ')
        for k in self.keyboard[10:21]:
            result.append(keys[k])
            result.append('   ')
        result.pop()
        result.append('\n   ')
        for k in self.keyboard[21:]:
            result.append(keys[k])
            result.append('   ')
        result.pop()
        return ''.join(result)


if __name__ == '__main__':
    initialize()

    import doctest
    doctest.testmod()

    kb = Keyboard()
    for i in range(100):
        kb = kb.optimize(i * 10_000)
        print(kb)
