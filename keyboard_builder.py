import math
import random

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
    'q',  'w',  'e',  'r',  't',  'y',  'u',  'i',  'o',  'p',
     'a',  's',  'd',  'f',  'g',  'h',  'j',  'k',  'l',  ';',  '"',
        'z',  'x',  'c',  'v',  'b',  'n',  'm',  '<',  '>',  '?'
    # 'q',  'y',  'o',  'k',  '.',  'c',  'w',  'u',  'f',  'p',
    #  'h',  'i',  'a',  't',  'd',  'x',  'r',  'e',  'n',  'b',  '.',
    #    '.',  '.',  '.',  'j',  'z',  's',  'm',  'l',  'g',  'v',
    # 'q',  'u',  'o',  'k',  '.',  '.',  'w',  'y',  '.',  'p',
    #  'h',  'e',  'a',  't',  'd',  'j',  'r',  'i',  'n',  'b',  '.',
    #    'z',  'x',  'c',  'f',  'v',  's',  'm',  'l',  'g',  '.',
    # 'y',  'k',  'o',  'w',  'j',  '.',  'p',  'u',  'd',  'f',
    #  'i',  'r',  'a',  'l',  'b',  'v',  't',  'e',  'n',  's',  'z',
    #    '.',  '.',  'h',  'g',  '.',  'c',  'm',  '.',  'x',  'q',
    # 'q',  'd',  'r',  'w',  'b',  'j',  'f',  'u',  'p',  '.',
    #  'a',  's',  'h',  't',  'g',  'y',  'n',  'e',  'o',  'i',  '.',
    #    'z',  'x',  'm',  'c',  'v',  'k',  'l',  '.',  '.',  '.',
    # """
]

effort = np.array([
    3.0,  1.5,  1.5,  2.0,  2.5,  3.0,  2.0,  1.5,  1.5,  2.5,
     2.0,  1.0,  1.0,  1.0,  2.0,  2.0,  1.0,  1.0,  1.0,  2.0,   3.1,
        4.0,  4.0,  1.5,  1.5,  3.0,  1.5,  1.5,  2.5,  3.0,  3.0,
], dtype=float)

keymap = {k: i for i, k in enumerate(keys)}

a = 1.0
b = 0.6
c = 0.5
roll_map = np.array([
    [  # q
        a, c, b, b, b, b, b, b, b, b,
        a, a, b, b, b, b, b, b, b, b, b,
        a, a, b, b, b, b, b, b, b, b
    ], [  # w
        a, a, c, b, b, b, b, b, b, b,
        b, a, b, b, b, b, b, b, b, b, b,
        a, a, b, b, b, b, b, b, b, b
    ], [  # e
        a, a, a, c, b, b, b, b, b, b,
        b, b, a, b, b, b, b, b, b, b, b,
        a, a, a, b, b, b, b, b, b, b
    ], [  # r
        a, a, a, a, a, b, b, b, b, b,
        b, b, a, a, a, b, b, b, b, b, b,
        a, a, a, a, a, b, b, b, b, b
    ], [  # t
        a, a, a, a, a, b, b, b, b, b,
        b, b, b, a, a, b, b, b, b, b, b,
        a, a, a, a, a, b, b, b, b, b
    ], [  # y
        a, a, a, a, a, a, a, b, b, b,
        b, b, b, b, b, a, a, b, b, b, b,
        b, b, b, b, b, a, a, a, a, a
    ], [  # u
        a, a, a, a, a, a, a, c, b, b,
        b, b, b, b, b, a, a, b, b, b, b,
        b, b, b, b, b, a, a, a, a, b
    ], [  # i
        a, a, a, a, a, a, a, a, c, b,
        b, b, b, b, b, b, b, a, b, b, b,
        b, b, b, b, b, b, b, a, b, b
    ], [  # o
        a, a, a, a, a, a, a, a, a, c,
        b, b, b, b, b, b, b, b, a, b, a,
        b, b, b, b, b, b, b, a, a, a
    ], [  # p
        a, a, a, a, a, a, a, a, a, a,
        b, b, b, b, b, b, b, b, b, a, a,
        b, b, b, b, b, b, b, a, a, a
    ], [  # a
        a, a, a, a, a, a, a, a, a, a,
        a, c, b, b, b, b, b, b, b, b, b,
        a, b, b, b, b, b, b, b, b, b
    ], [  # s
        a, a, a, a, a, a, a, a, a, a,
        a, a, c, b, b, b, b, b, b, b, b,
        a, a, b, b, b, b, b, b, b, b
    ], [  # d
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, c, b, b, b, b, b, b, b,
        b, a, a, b, b, b, b, b, b, b
    ], [  # f
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, b, b, b, b, b, b,
        b, a, a, a, a, b, b, b, b, b
    ], [  # g
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, b, b, b, b, b, b,
        b, a, a, a, a, b, b, b, b, b
    ], [  # h
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, b, b, b, b,
        b, b, b, b, b, a, a, a, a, b
    ], [  # j
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, c, b, b, b,
        b, b, b, b, b, a, a, b, b, b
    ], [  # k
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, c, b, b,
        b, b, b, b, b, b, b, a, b, a
    ], [  # l
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, c, b,
        b, b, b, b, b, b, b, a, a, a
    ], [  # ;
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        b, b, b, b, b, b, b, b, b, a
    ], [  # '
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        b, b, b, b, b, b, b, b, a, a
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
        a, a, a, a, a, b, b, b, b, b
    ], [  # v
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, b, b, b, b, b
    ], [  # b
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, b, b, b, b, b
    ], [  # n
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, b, b, b
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

letter_frequency: np.ndarray = np.zeros(len(keys), dtype=float)
digram_frequency: np.ndarray = np.zeros((len(keys), len(keys)), dtype=float)


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
                digram_frequency[keymap[l0], keymap[l1]] = float(f)

    def initialize_roll_map():
        i = np.tril_indices_from(roll_map)
        roll_map[i] = roll_map.T[i]

    initialize_letters()
    initialize_digrams()
    initialize_roll_map()


class Keyboard:
    def __init__(self, other: 'Keyboard' = None):
        if other:
            self.keyboard = other.keyboard.copy()
            self.penalty = other.penalty
        else:
            self.keyboard = np.array(range(len(keys)), dtype=int)
            self.calculate_penalties()

    def neighbor(self):
        """
        >>> kb = Keyboard()
        >>> for i in range(1000):
        ...     n0, n1 = kb.neighbor()
        ...     assert n0 != n1
        """
        key1letter = ''
        key2letter = ''
        i1 = -1
        i2 = -1
        while i1 == -1 or {key1letter, key2letter} & set(''):
            i1 = random.randint(1, len(self.keyboard) - 1)
            i2 = random.randint(1, len(self.keyboard) - 1)
            if i1 == i2:
                i2 = 0
            key1letter = keys[self.keyboard[i1]]
            key2letter = keys[self.keyboard[i2]]

        return i1, i2

    def calculate_penalties(self):
        # letter
        kb = self.keyboard
        letter_delta = np.sum(letter_frequency[kb] * effort)

        # digram
        digram_delta = np.sum((effort[None, kb] + effort[kb, None])
                              * roll_map
                              * digram_frequency[kb][:, kb])

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
        self.keyboard[i1] = key2
        self.keyboard[i2] = key1

    def optimize(self, iterations: int):
        best = self.deepcopy()
        curr = self.deepcopy()
        for i in range(iterations):
            if i % (iterations // 100) == 0:
                print(end='.')
            t = math.exp((iterations - i - 1) / iterations) - 1
            prev = curr.deepcopy()
            i1, i2 = curr.neighbor()
            curr.score_and_apply_neighbor(i1, i2)
            if curr.penalty < prev.penalty + t:
                # We're good
                if curr.penalty < best.penalty:
                    best = curr.deepcopy()
            else:
                # Undo
                curr = prev

        print('\n', curr.penalty, best.penalty)
        return best

    def deepcopy(self):
        return Keyboard(self)

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
        kb = kb.optimize(i * 100_000)
        print(kb)
