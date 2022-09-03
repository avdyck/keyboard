import random

import numpy as np

qwerty = [
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ':',
    'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/',
    '?', ' ',
]

ytrewq = [
    'p', 'o', 'i', 'u', 'y', 't', 'r', 'e', 'w', 'q',
    ':', 'l', 'k', 'j', 'h', 'g', 'f', 'd', 's', 'a',
    '/', '.', ',', 'm', 'n', 'b', 'v', 'c', 'x', 'z',
    '?', ' ',
]

custom1 = [
    'x', 'p', 'u', 'o', 'q', 'f', 'l', 'r', 'd', 'y',
    'k', 'i', 'e', 'a', 'b', 'm', 'n', 't', 's', 'c',
    'z', '.', '.', '.', '.', 'v', 'h', 'w', 'g', 'j',
    '?', ' '
]

custom2 = [
    'x', 'm', 'g', 'b', 'z', 'q', 'f', 'u', 'o', 'y',
    'r', 'l', 's', 't', 'p', 'k', 'n', 'e', 'a', 'i',
    'c', 'j', 'w', 'd', 'v', ':', 'h', ',', '.', '/',
    '?', ' '
]

hand = [
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    2, 2,
]

fingers = [
    0, 1, 2, 3, 3, 3, 3, 2, 1, 0,
    0, 1, 2, 3, 3, 3, 3, 2, 1, 0,
    0, 1, 2, 3, 3, 3, 3, 2, 1, 0,
    8, 8,
]

rows = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3,
]

keymap = {k: i for i, k in enumerate(qwerty)}

fixed_keys = set(keymap[k] for k in ' ?')

effort = np.array([
    5.0, 1.1, 1.0, 2.0, 3.0, 0, 0, 0, 0, 0,
    0.8, 0.7, 0.6, 0.5, 1.8, 0, 0, 0, 0, 0,
    1.6, 2.2, 2.0, 1.0, 3.0, 0, 0, 0, 0, 0,
    0.0, 0.0,
], dtype=float)

# effort = np.array([
#     9.9, 1.0, 1.0, 1.0, 5.0, 0, 0, 0, 0, 0,
#     1.0, 0.6, 0.5, 0.4, 1.5, 0, 0, 0, 0, 0,
#     2.5, 2.0, 5.0, 1.5, 7.0, 0, 0, 0, 0, 0,
#     0, 0
# ], dtype=float)

# effort = np.array([
#     3.0, 2.4, 2.0, 2.2, 3.2, 3.2, 2.0, 2.0, 2.4, 3.0,
#     1.6, 1.3, 1.1, 1.0, 2.9, 2.9, 1.1, 1.1, 1.3, 1.6,
#     3.2, 2.6, 2.4, 1.6, 3.0, 3.0, 1.6, 2.4, 2.6, 3.2,
# ], dtype=float)


digram_effort = np.zeros((len(qwerty), len(qwerty)), dtype=float)
trigram_effort = np.zeros((len(qwerty), len(qwerty), len(qwerty)), dtype=float)

a = 0.3
b = 0.4
c = 0.6
d = 1.0
e = 1.5
f = 2.0

m = 2.0
n = 3.0
o = 4.0
p = o

z = 0.5
roll_map = np.array([
    [  # q
        m, a, a, a, a, z, z, z, z, z,
        n, d, d, c, b, z, z, z, z, z,
        o, f, f, d, d, z, z, z, z, z,
        z, z,
    ], [  # w
        0, m, a, a, a, z, z, z, z, z,
        c, n, c, a, b, z, z, z, z, z,
        f, o, f, c, d, z, z, z, z, z,
        z, z,
    ], [  # e
        0, 0, m, a, b, z, z, z, z, z,
        b, d, n, a, c, z, z, z, z, z,
        d, f, o, c, d, z, z, z, z, z,
        z, z,
    ], [  # r
        0, 0, 0, m, n, z, z, z, z, z,
        b, c, d, n, o, z, z, z, z, z,
        c, d, e, o, p, z, z, z, z, z,
        z, z,
    ], [  # t
        0, 0, 0, 0, m, z, z, z, z, z,
        b, c, c, o, n, z, z, z, z, z,
        c, d, e, p, o, z, z, z, z, z,
        z, z,
    ], [  # y
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # u
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # i
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # o
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # p
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # a
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        m, a, a, a, a, z, z, z, z, z,
        n, d, c, b, c, z, z, z, z, z,
        z, z,
    ], [  # s
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, m, a, a, c, z, z, z, z, z,
        c, n, c, b, c, z, z, z, z, z,
        z, z,
    ], [  # d
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, m, a, c, z, z, z, z, z,
        c, d, n, b, c, z, z, z, z, z,
        z, z,
    ], [  # f
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, m, n, z, z, z, z, z,
        b, c, d, n, o, z, z, z, z, z,
        z, z,
    ], [  # g
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, m, z, z, z, z, z,
        c, d, d, o, n, z, z, z, z, z,
        z, z,
    ], [  # h
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # j
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # k
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # l
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # ;
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z, z, z, z, 0, 0, 0, 0, 0,
        z, z,
    ], [  # z
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        m, a, a, a, b, z, z, z, z, z,
        z, z,
    ], [  # x
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, m, a, b, d, z, z, z, z, z,
        z, z,
    ], [  # c
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, m, a, b, z, z, z, z, z,
        z, z,
    ], [  # v
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, m, n, z, z, z, z, z,
        z, z,
    ], [  # b
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, m, z, z, z, z, z,
        z, z,
    ], [  # n
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z,
    ], [  # m
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z,
    ], [  # ,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z,
    ], [  # .
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z,
    ], [  # /
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        z, z,
    ], [  # spc
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        m, z,
    ], [  # spc
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, m,
    ]
], dtype=float)

letter_frequency: np.ndarray = np.zeros(len(qwerty), dtype=float)
digram_frequency: np.ndarray = np.zeros((len(qwerty), len(qwerty)), dtype=float)
trigram_frequency: np.ndarray = np.zeros((len(qwerty), len(qwerty), len(qwerty)), dtype=float)

def initialize():
    def initialize_letters():
        with open('freq-letter.txt') as fh:
            for line in fh:
                l, f = line.strip().split('\t')
                letter_frequency[keymap[l]] = float(f)

    def initialize_digrams():
        with open('freq-digram.txt') as fh:
            for line in fh:
                (l0, l1), f = line.replace('\n', '').split('\t')
                digram_frequency[keymap[l0], keymap[l1]] = float(f)

    def initialize_trigrams():
        with open('freq-trigram.txt') as fh:
            for line in fh:
                (l0, l1, l2), f = line.replace('\n', '').split('\t')
                trigram_frequency[keymap[l0], keymap[l1], keymap[l2]] = float(f)

    def initialize_roll_map():
        for i0, k0 in enumerate(qwerty):
            for i1, k1 in enumerate(qwerty):
                if i1 < i0:
                    assert roll_map[i0][i1] == 0
                    roll_map[i0][i1] = roll_map[i1][i0]

        for i0, k0 in enumerate(qwerty):
            mirror_i0 = ytrewq.index(k0)
            if hand[i0] == 0:
                effort[mirror_i0] = effort[i0]

                for i1, k1 in enumerate(qwerty):
                    mirror_i1 = ytrewq.index(k1)

                    if hand[i1] == 0:
                        roll_map[mirror_i0][mirror_i1] = roll_map[i0][i1]

        assert roll_map[keymap['p'], keymap['k']] == \
               roll_map[keymap['q'], keymap['d']] == \
               roll_map[keymap['k'], keymap['p']] == \
               roll_map[keymap['d'], keymap['q']]

        # # favor inward rolls slightly
        # for i0 in range(len(qwerty)):
        #     for i1 in range(len(qwerty)):
        #         if hand[i0] == hand[i1] and fingers[i0] < fingers[i1]:
        #             roll_map[i0][i1] *= 0.75

        for i0 in range(len(qwerty)):
            for i1 in range(len(qwerty)):
                digram_effort[i0][i1] = roll_map[i0][i1] * (effort[i0] + effort[i1])

    def initialize_tri_roll_map():

        for i0 in range(len(qwerty)):
            for i1 in range(len(qwerty)):
                for i2 in range(len(qwerty)):

                    tri_effort = digram_effort[i0, i1] + digram_effort[i1, i2] + 0.5 * digram_effort[i0, i2]

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
            self.keyboard = np.array(range(len(qwerty)), dtype=int)
            # np.random.shuffle(self.keyboard)

            changed = True
            while changed:
                changed = False
                for i, k in enumerate(self.keyboard):
                    if i != k and k in fixed_keys:
                        self.apply_neighbor(i, k)
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
        candidates = [i1 for i1, k in enumerate(self.keyboard)
                      if k not in fixed_keys
                      if i1 != i0]

        i1 = random.choice(candidates)
        return i0, i1

    def calculate_penalties(self):
        # letter
        kb = self.keyboard

        self.penalty = 0 \
                       + np.sum(letter_frequency[kb] * effort) \
                       + np.sum(digram_effort * digram_frequency[kb][:, kb]) \
                       + np.sum(trigram_effort * trigram_frequency[kb][:, kb][:, :, kb])

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

            threshold = curr.penalty * 1.04
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
            result.append(qwerty[k])
            result.append('   ')
        result.pop()
        result.append('\n')
        for k in self.keyboard[10:20]:
            result.append(qwerty[k])
            result.append('   ')
        result.pop()
        result.append('\n')
        for k in self.keyboard[20:30]:
            result.append(qwerty[k])
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
