import math

import numpy as np

qwerty = [
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
     'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '"',
        'z', 'x', 'c', 'v', 'b', 'n', 'm', '<', '>', '?'
]

digram_only_optimized = [
    'y', 'p', 'm', 'k', 'b', 'z', 'g', 'u', 'o', 'b',
     'g', 'i', 't', 's', 'f', 'j', 'n', 'e', 'a', 'c', '"',
        'z', 'x', 'd', 'h', ';', 'r', 'l', '<', '>', '?',
]

trigram_optimized = [
    'y',  'p',  'o',  'f',  'b',  'x',  'm',  'u',  'w',  'j',
     'g',  'i',  'a',  't',  'k',  'v',  'n',  'e',  'r',  'c',  '"',
       ';',  'q',  'd',  's',  'z',  'l',  'h',  '<',  '>',  '?',
]

keys = trigram_optimized
fixed_keys = set('"<>?;')

keymap = {k: i for i, k in enumerate(keys)}

effort = np.array([
    2.5, 2.0, 1.5, 2.0, 2.5, 3.0, 2.0, 1.5, 2.0, 2.5,
    1.8, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.8, 3.0,
    3.0, 3.0, 1.5, 1.5, 3.0, 1.5, 1.5, 3.0, 3.0, 4.0,
], dtype=float)

digram_effort = np.zeros((len(keys), len(keys)), dtype=float)
trigram_effort = np.zeros((len(keys), len(keys), len(keys)), dtype=float)

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
        b, a, b, b, b, b, b, b, b, b
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
], dtype=float)

letter_frequency: np.ndarray = np.zeros(len(keys), dtype=float)
digram_frequency: np.ndarray = np.zeros((len(keys), len(keys)), dtype=float)
trigram_frequency: np.ndarray = np.zeros((len(keys), len(keys), len(keys)), dtype=float)


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

    def initialize_trigrams():
        with open('freq-trigram.txt') as fh:
            for line in fh:
                (l0, l1, l2), f = line.strip().split('\t')
                trigram_frequency[keymap[l0], keymap[l1], keymap[l2]] = float(f)

    def initialize_roll_map():
        i = np.tril_indices_from(roll_map)
        roll_map[i] = roll_map.T[i]
        for i0 in range(len(keys)):
            for i1 in range(len(keys)):
                digram_effort[i0][i1] = roll_map[i0][i1] * (effort[i0] + effort[i1]) / 2

    def initialize_tri_roll_map():

        cols = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        ]

        rows = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        ]

        hand = [
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        ]

        for i0 in range(len(keys)):
            for i1 in range(len(keys)):
                for i2 in range(len(keys)):
                    # Best case scenario: average of digram efforts
                    trigram_effort[i0, i1, i2] = (2 * digram_effort[i0, i1] + 2 * digram_effort[i1, i2] + digram_effort[i0, i2]) / 5

                    if hand[i0] == hand[i1] == hand[i2] and i0 != i2:
                        horizontal = (cols[i0] < cols[i1] < cols[i2]) or (cols[i0] > cols[i1] > cols[i2])
                        vertical = (rows[i0] <= rows[i1] <= rows[i2]) or (rows[i0] >= rows[i1] >= rows[i2])
                        if not horizontal or not vertical:
                            trigram_effort[i0, i1, i2] *= min(digram_effort[i0, i1] + effort[i2], effort[i0] + digram_effort[i1, i2])

    initialize_letters()
    initialize_digrams()
    initialize_trigrams()
    initialize_roll_map()
    initialize_tri_roll_map()


class Keyboard:
    def __init__(self, other: 'Keyboard' = None):
        if other:
            self.keyboard = other.keyboard.copy()
            self.penalty = other.penalty
        else:
            self.keyboard = np.array(range(len(keys)), dtype=int)
            # np.random.shuffle(self.keyboard)
            self.calculate_penalties()

    def neighbor(self):
        """
        >>> kb = Keyboard()
        >>> for i in range(1000):
        ...     n0, n1 = kb.neighbor()
        ...     assert n0 != n1
        """
        global fixed_keys
        candidates = [i for i, k in enumerate(self.keyboard) if keys[k] not in fixed_keys]

        return np.random.choice(candidates, 2, replace=False)

    def calculate_penalties(self):
        # letter
        kb = self.keyboard
        letter_delta = np.sum(letter_frequency[kb] * effort)

        # digram
        digram_delta = np.sum(digram_effort * digram_frequency[kb][:, kb]) / 2

        # trigram
        trigram_delta = np.sum(trigram_effort * trigram_frequency[kb][:, kb][:, :, kb]) / 3

        self.penalty = letter_delta + digram_delta + trigram_delta

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
            t = math.exp((iterations - i - 1) / iterations) - 1
            prev_penalty = curr.penalty
            i1, i2 = curr.neighbor()
            curr.score_and_apply_neighbor(i1, i2)
            if curr.penalty < prev_penalty + t:
                # We're good
                if curr.penalty < best.penalty:
                    print(f'{curr}\n{best.penalty} -> {curr.penalty}')
                    best = curr.deepcopy()
            else:
                # Undo
                curr.apply_neighbor(i1, i2)
                curr.penalty = prev_penalty

        return best

    def deepcopy(self):
        return Keyboard(self)

    def __repr__(self):
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
    print(kb)
    print(kb.penalty)
    # import cProfile
    # cProfile.run('kb.optimize(100_000)')
    reheats = 1_000_000
    its = 500
    for i in range(reheats):
        kb = kb.optimize(its - i * its // reheats)

    for i in range(its):
        kb = kb.optimize(its - i)
