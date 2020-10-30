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

trigram_optimized_1 = [
   'x',  'p',  'o',  'k',  'v',  'f',  'm',  'u',  'c',  'z',
    'j',  'i',  'a',  't',  'g',  'w',  'n',  'e',  'r',  'y',  '"',
      ';',  'q',  'd',  's',  'b',  'l',  'h',  '<',  '>',  '?',
]

trigram_optimized_2 = [
   'q',  'w',  'o',  'p',  'b',  'f',  'l',  'u',  's',  ';',
    'j',  'i',  'a',  't',  'g',  'k',  'n',  'e',  'r',  'y',  '"',
      'z',  'x',  'c',  'd',  'v',  'h',  'm',  '<',  '>',  '?',
]

keys = trigram_optimized_2
keymap = {k: i for i, k in enumerate(keys)}

fixed_keys = set(keymap[k] for k in '<>?";oiaueqwzxcvgsrlm')

effort = np.array([
    6.0, 1.6, 1.3, 1.5, 2.0, 2.6, 1.5, 1.3, 1.6, 6.0,
     5.0, 1.1, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.1, 5.0, 8.0,
        6.0, 8.0, 1.2, 1.2, 2.5, 1.2, 1.2, 1.8, 2.2, 8.0,
], dtype=float)

digram_effort = np.zeros((len(keys), len(keys)), dtype=float)
trigram_effort = np.zeros((len(keys), len(keys), len(keys)), dtype=float)

a = 1.0
b = 0.6
c = 0.6
d = 0.5
roll_map = np.array([
    [  # q
        a, d, b, b, b, c, c, c, c, c,
        a, a, b, b, b, c, c, c, c, c, c,
        a, a, b, b, b, c, c, c, c, c
    ], [  # w
        a, a, d, b, b, c, c, c, c, c,
        b, a, b, b, b, c, c, c, c, c, c,
        a, a, b, b, b, c, c, c, c, c
    ], [  # e
        a, a, a, d, b, c, c, c, c, c,
        b, b, a, b, b, c, c, c, c, c, c,
        a, a, a, b, b, c, c, c, c, c
    ], [  # r
        a, a, a, a, a, c, c, c, c, c,
        b, b, a, a, a, c, c, c, c, c, c,
        a, a, a, a, a, c, c, c, c, c
    ], [  # t
        a, a, a, a, a, c, c, c, c, c,
        b, b, b, a, a, c, c, c, c, c, c,
        a, a, a, a, a, c, c, c, c, c
    ], [  # y
        a, a, a, a, a, a, a, b, b, b,
        c, c, c, c, c, a, a, b, b, b, b,
        c, c, c, c, c, a, a, a, a, a
    ], [  # u
        a, a, a, a, a, a, a, d, b, b,
        c, c, c, c, c, a, a, b, b, b, b,
        c, c, c, c, c, a, a, a, a, b
    ], [  # i
        a, a, a, a, a, a, a, a, d, b,
        c, c, c, c, c, b, b, a, b, b, b,
        c, c, c, c, c, b, b, a, b, b
    ], [  # o
        a, a, a, a, a, a, a, a, a, d,
        c, c, c, c, c, b, b, b, a, b, a,
        c, c, c, c, c, b, b, a, a, a
    ], [  # p
        a, a, a, a, a, a, a, a, a, a,
        c, c, c, c, c, b, b, b, b, a, a,
        c, c, c, c, c, b, b, a, a, a
    ], [  # a
        a, a, a, a, a, a, a, a, a, a,
        a, d, b, b, b, c, c, c, c, c, c,
        a, b, b, b, b, c, c, c, c, c
    ], [  # s
        a, a, a, a, a, a, a, a, a, a,
        a, a, d, b, b, c, c, c, c, c, c,
        a, a, b, b, b, c, c, c, c, c
    ], [  # d
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, d, b, c, c, c, c, c, c,
        b, a, b, b, b, c, c, c, c, c
    ], [  # f
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, c, c, c, c, c, c,
        b, a, a, a, a, c, c, c, c, c
    ], [  # g
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, c, c, c, c, c, c,
        b, a, a, a, a, c, c, c, c, c
    ], [  # h
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, b, b, b, b,
        c, c, c, c, c, a, a, a, a, b
    ], [  # j
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, d, b, b, b,
        c, c, c, c, c, a, a, b, b, b
    ], [  # k
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, d, b, b,
        c, c, c, c, c, b, b, a, b, a
    ], [  # l
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, d, b,
        c, c, c, c, c, b, b, a, a, a
    ], [  # ;
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        c, c, c, c, c, b, b, b, b, a
    ], [  # '
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        c, c, c, c, c, b, b, b, a, a
    ], [  # z
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, d, b, b, b, c, c, c, c, c
    ], [  # x
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, d, b, b, c, c, c, c, c
    ], [  # c
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, c, c, c, c, c
    ], [  # v
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, c, c, c, c, c
    ], [  # b
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, c, c, c, c, c
    ], [  # n
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, b, b, b
    ], [  # m
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, d, b, b
    ], [  # ,
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, d, b
    ], [  # .
        a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, a, a,
        a, a, a, a, a, a, a, a, a, d
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
                digram_effort[i0][i1] = roll_map[i0][i1] * (effort[i0] + effort[i1])

    def initialize_tri_roll_map():

        fingers = [
            0, 1, 2, 3, 3, 4, 4, 5, 6, 7,
            0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 7,
            0, 1, 3, 3, 3, 4, 4, 5, 6, 7,
        ]

        fongers = [
            0, 1, 2, 3, 3, 3, 3, 2, 1, 0,
            0, 1, 2, 3, 3, 3, 3, 2, 1, 0, 0,
            0, 1, 3, 3, 3, 3, 3, 2, 1, 0,
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

                    if i0 == i1 == i2:
                        # who cares lol
                        tri_effort = 3 * effort[i1]

                    elif i0 == i2:
                        if roll_map[i0, i1] == a or fongers[i0] == 0:
                            # bad case where i1 resets i0 progress
                            tri_effort = 2 * effort[i0] + effort[i1]
                        else:
                            tri_effort = digram_effort[i1, i2]

                    elif i0 == i1:
                        tri_effort = effort[i0] + digram_effort[i1, i2]
                    elif i1 == i2:
                        tri_effort = digram_effort[i0, i1] + effort[i2]
                    elif hand[i0] != hand[i1] == hand[i2]:
                        tri_effort = digram_effort[i0, i1] + digram_effort[i1, i2]
                    elif hand[i0] == hand[i1] != hand[i2]:
                        tri_effort = digram_effort[i0, i1] + digram_effort[i1, i2]
                    elif hand[i0] != hand[i1] != hand[i2]:
                        tri_effort = digram_effort[i0, i1] + digram_effort[i1, i2] + 0.5 * digram_effort[i0, i2]

                    elif hand[i0] == hand[i1] == hand[i2]:
                        # all same hand: could be worst or best case scenario depending on which fingeys are used
                        finger0 = fingers[i0]
                        finger1 = fingers[i1]
                        finger2 = fingers[i2]
                        horizontal_roll = (finger0 < finger1 < finger2) or (finger0 > finger1 > finger2)

                        if horizontal_roll or finger0 == finger2:
                            tri_effort = digram_effort[i0, i1] + digram_effort[i1, i2] + 0.5 * digram_effort[i0, i2]
                        else:
                            # worst case: same hand non horizontal weird shit: not even gonna look at the roll graph
                            tri_effort = effort[i0] + effort[i1] + effort[i2]
                    else:
                        raise Exception('huh')

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

            self.calculate_penalties()

    def neighbor(self):
        """
        >>> kb = Keyboard()
        >>> for i in range(1000):
        ...     n0, n1 = kb.neighbor()
        ...     assert n0 != n1
        """
        global fixed_keys
        candidates = [i for i, k in enumerate(self.keyboard) if k not in fixed_keys]

        return np.random.choice(candidates, 2, replace=False)

    def calculate_penalties(self):
        # letter
        kb = self.keyboard
        letter_delta = np.sum(letter_frequency[kb] * effort)

        # digram
        digram_delta = np.sum(digram_effort * digram_frequency[kb][:, kb])

        # trigram
        trigram_delta = np.sum(trigram_effort * trigram_frequency[kb][:, kb][:, :, kb])

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
    reheats = 100_000
    its = 200
    for i in range(reheats):
        kb = kb.optimize(its - i * its // reheats)

    for i in range(its):
        kb = kb.optimize(100)
