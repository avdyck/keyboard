import math
import random

import numpy as np

qwerty = [
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',
    'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'
]

optimized = [
    'q', 'p', 'u', 'o', 'x', 'f', 'l', 'r', 'd', 'b',
    'c', 'i', 'e', 'a', 'g', 'v', 'n', 't', 's', 'w',
    'z', ';', '<', '>', '?', 'j', 'h', 'm', 'k', 'y',
]

custom1 = [
    'v', 'p', 'u', 'o', 'q', 'f', 'l', 'r', 'd', 'y',
    'k', 'i', 'e', 'a', 'j', 'm', 'n', 't', 's', 'c',
    'z', '.', '.', '.', '.', 'x', 'h', 'w', 'g', 'b',
]

custom2 = [
    'y', 'o', 'u', 'g', 'q', 'w', 'c', 'l', 'h', 'z',
    'i', 'a', 'e', 't', 'k', 'f', 'd', 'r', 'n', 's',
    '.', '.', '.', 'x', '.', 'b', 'p', 'j', 'm', 'v',
]

bruh_keyboard = [
    'b', 'r', 'u', 'h', 'x', 'f', 'd', 'o', 'p', 'y',
    'c', 's', 'e', 'n', 'l', 'g', 't', 'a', 'i', 'w',
    'q', 'j', ';', 'm', 'z', 'v', 'k', '<', '>', '?',
]

hand = [
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
]

fingers = [
    0, 1, 2, 3, 3, 4, 4, 5, 6, 7,
    0, 1, 2, 3, 3, 4, 4, 5, 6, 7,
    0, 1, 3, 3, 3, 4, 4, 5, 6, 7,
]

rows = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
]

keys = qwerty
keymap = {k: i for i, k in enumerate(keys)}
qwertymap = {k: i for i, k in enumerate(qwerty)}

fixed_keys = set(keymap[k] for k in './,qzxcv')
vim_keys = set(qwertymap[k] for k in '')
vimmable_keys = set(keymap[k] for k in 'hjkl')

effort = np.array([
    2.0, 1.1, 1.05, 1.0, 1.9, 1.9, 1.0, 1.05, 1.1, 2.0,
    1.0, 0.7, 0.65, 0.6, 1.5, 1.5, 0.6, 0.65, 0.7, 1.0,
    1.8, 1.4, 1.35, 1.3, 1.7, 1.7, 1.3, 1.35, 1.4, 1.8,
], dtype=float)

# effort = np.array([
#     9.9, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0, 9.9,
#     5.0, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5, 5.0,
#     7.0, 2.0, 5.0, 1.5, 7.0, 7.0, 1.5, 5.0, 2.0, 7.0,
# ], dtype=float)

# effort = np.array([
#     3.0, 2.4, 2.0, 2.2, 3.2, 3.2, 2.0, 2.0, 2.4, 3.0,
#     1.6, 1.3, 1.1, 1.0, 2.9, 2.9, 1.1, 1.1, 1.3, 1.6,
#     3.2, 2.6, 2.4, 1.6, 3.0, 3.0, 1.6, 2.4, 2.6, 3.2,
# ], dtype=float)


digram_effort = np.zeros((len(keys), len(keys)), dtype=float)
trigram_effort = np.zeros((len(keys), len(keys), len(keys)), dtype=float)

a = 0.3
b = 0.4
c = 0.5
d = 0.7

e = 2.0
f = 2.2
g = 2.4
h = 2.6

z = 0.5
roll_map = np.array([
    [  # q
        e, a, b, b, c, z, z, z, z, z,
        f, d, d, c, c, z, z, z, z, z,
        g, g, g, d, d, z, z, z, z, z,
    ], [  # w
        0, e, a, b, b, z, z, z, z, z,
        c, f, c, b, c, z, z, z, z, z,
        g, g, g, c, d, z, z, z, z, z,
    ], [  # e
        0, 0, e, a, b, z, z, z, z, z,
        c, d, f, b, b, z, z, z, z, z,
        f, g, g, c, c, z, z, z, z, z,
    ], [  # r
        0, 0, 0, e, f, z, z, z, z, z,
        b, b, c, f, g, z, z, z, z, z,
        c, f, g, g, h, z, z, z, z, z,
    ], [  # t
        0, 0, 0, 0, e, z, z, z, z, z,
        b, c, c, g, f, z, z, z, z, z,
        c, f, g, h, g, z, z, z, z, z,
    ], [  # y
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # u
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # i
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # o
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # p
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # a
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        e, a, b, b, b, z, z, z, z, z,
        f, d, d, c, c, z, z, z, z, z,
    ], [  # s
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, e, a, b, c, z, z, z, z, z,
        1, f, d, c, c, z, z, z, z, z,
    ], [  # d
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, e, a, b, z, z, z, z, z,
        c, d, f, b, c, z, z, z, z, z,
    ], [  # f
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, e, f, z, z, z, z, z,
        c, c, d, f, g, z, z, z, z, z,
    ], [  # g
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, e, z, z, z, z, z,
        c, d, d, g, f, z, z, z, z, z,
    ], [  # h
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # j
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # k
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # l
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # ;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
    ], [  # z
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        e, a, b, b, c, z, z, z, z, z,
    ], [  # x
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, e, a, b, c, z, z, z, z, z,
    ], [  # c
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, e, a, b, z, z, z, z, z,
    ], [  # v
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, e, f, z, z, z, z, z,
    ], [  # b
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, e, z, z, z, z, z,
    ], [  # n
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ], [  # m
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ], [  # ,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ], [  # .
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ], [  # /
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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

        for i in range(3):
            for l in range(3):
                for j, k in ((0, 9), (1, 8), (2, 7), (3, 6), (4, 5)):
                    roll_map[k + 10 * l][5 + 10 * i:10 + 10 * i] = roll_map[j + 10 * l][0 + 10 * i:5 + 10 * i][::-1]

        tmp_keymap = {k: i for i, k in enumerate(qwerty)}
        assert roll_map[tmp_keymap['p'], tmp_keymap['k']] == roll_map[tmp_keymap['q'], tmp_keymap['d']]
        assert roll_map[tmp_keymap['v'], tmp_keymap['e']] == roll_map[tmp_keymap['m'], tmp_keymap['i']]
        assert roll_map[tmp_keymap['y'], tmp_keymap['p']] == roll_map[tmp_keymap['t'], tmp_keymap['q']]

        for i0 in range(len(keys)):
            for i1 in range(len(keys)):
                digram_effort[i0][i1] = roll_map[i0][i1] * (effort[i0] + effort[i1])

    def initialize_tri_roll_map():

        for i0 in range(len(keys)):
            for i1 in range(len(keys)):
                for i2 in range(len(keys)):

                    tri_effort = (0.4 * digram_effort[i0, i1] + 0.4 * digram_effort[i1, i2] + 0.2 * digram_effort[i0, i2])

                    # hand pingpong good case
                    if hand[i0] == hand[i1] == hand[i2]:
                        finger0 = fingers[i0]
                        finger1 = fingers[i1]
                        finger2 = fingers[i2]
                        row0 = rows[i0]
                        row1 = rows[i1]
                        row2 = rows[i2]

                        nice_horizontal_roll = (finger0 >= finger1 >= finger2) or (finger0 <= finger1 <= finger2)
                        nice_vertical_roll = (row0 <= row1 <= row2) or (row0 >= row1 >= row2)
                        if not nice_horizontal_roll or not nice_vertical_roll:
                            tri_effort *= 2

                    trigram_effort[i0, i1, i2] = tri_effort

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

            changed = True
            while changed:
                changed = False
                for i, k in enumerate(self.keyboard):
                    if i != k and k in fixed_keys:
                        self.apply_neighbor(i, k)
                        changed = True

                should_be_vim_keys = []
                for i in vim_keys:
                    if self.keyboard[i] not in vimmable_keys:
                        should_be_vim_keys.append(i)

                for i, k in enumerate(self.keyboard):
                    if should_be_vim_keys and k in vimmable_keys and i not in vim_keys:
                        self.apply_neighbor(i, should_be_vim_keys.pop())
                        changed = True

            self.calculate_penalties()

    def neighbor(self):
        """
        >>> kb = Keyboard()
        >>> for i in range(1000):
        ...     n0, n1 = kb.neighbor()
        ...     assert n0 != n1
        """
        global fixed_keys

        candidates = [i for i, k in enumerate(self.keyboard)
                      if k not in fixed_keys]

        i0 = random.choice(candidates)

        if i0 in vim_keys:
            candidates = [i1 for i1, k in enumerate(self.keyboard)
                          if i1 != i0
                          if k in vimmable_keys]

        else:
            candidates = [i1 for i1, k in enumerate(self.keyboard)
                          if k not in fixed_keys
                          if i1 not in vim_keys
                          if i1 != i0]

        i1 = random.choice(candidates)
        if self.keyboard[i0] not in vimmable_keys and i1 in vim_keys:
            raise Exception

        return i0, i1

    def calculate_penalties(self):
        # letter
        kb = self.keyboard
        # letter_delta = np.sum(letter_frequency[kb] * effort)

        # digram
        # digram_delta = np.sum(digram_effort * digram_frequency[kb][:, kb])

        # trigram
        trigram_delta = np.sum(trigram_effort * trigram_frequency[kb][:, kb][:, :, kb])

        self.penalty = trigram_delta

    def score_and_apply_neighbor(self, i1: int, i2: int):
        self.apply_neighbor(i1, i2)
        self.calculate_penalties()

    def apply_neighbor(self, i1: int, i2: int) -> None:
        key1 = self.keyboard[i1]
        key2 = self.keyboard[i2]
        self.keyboard[i1] = key2
        self.keyboard[i2] = key1

    def optimize(self):
        best = self.deepcopy()
        curr = self.deepcopy()

        for i in range(100):
            max_idle = max_accept = (1 + i) * 100

            threshold = curr.penalty * 1.02
            accept = 0
            idle = 0
            while idle < max_idle:
                prev_penalty = curr.penalty
                i1, i2 = curr.neighbor()
                curr.score_and_apply_neighbor(i1, i2)
                if curr.penalty >= threshold:
                    # Undo
                    idle += 1
                    curr.apply_neighbor(i1, i2)
                    curr.penalty = prev_penalty
                else:
                    # Bingo
                    if curr.penalty < best.penalty:
                        print(f'{curr}\n{best.penalty} -> {curr.penalty}')
                        best = curr.deepcopy()

                    idle = 0
                    accept += 1
                    if accept > max_accept:
                        accept = 0
                        threshold = curr.penalty

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
        result.append('\n')
        for k in self.keyboard[10:20]:
            result.append(keys[k])
            result.append('   ')
        result.pop()
        result.append('\n')
        for k in self.keyboard[20:]:
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

    kb.optimize()

    # # import cProfile
    # # cProfile.run('kb.optimize(100_000)')
    # reheats = 100_000
    # its = 100
    # for i in range(reheats):
    #     kb = kb.optimize(its)
