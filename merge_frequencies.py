from collections import defaultdict

word_freqs = []


def parse_file(filename, result, multiplier = 1.0):
    with open(filename) as fh:
        for line in fh:
            stripped = line.lower().strip().replace("\t", " ")
            if stripped:
                split = stripped.split(" ")
                if len(split) > 1:
                    result.append((float(split[-1]) * multiplier, split[0]))


parse_file("english.txt", word_freqs)
parse_file("dutch.txt", word_freqs, 0.5)
# parse_file("code.txt", word_freqs, 1)
# parse_file("vim.txt", word_freqs)
word_freqs.sort(reverse=True)

s = sum(f for f, w in word_freqs)

with open("freq-word.txt", 'w') as fh:
    for f, w in word_freqs:
        fh.write(f'{w}\t{f}\t\n')


def write_freqs(freqs, filename):
    freqs = [(f, l) for l, f in freqs.items()]
    freqs.sort(reverse=True)
    s = sum(f for f, w in freqs)
    with open(filename, 'w') as fh:
        for f, l in freqs:
            fh.write(f'{l}\t{f / s}\n')


letter_freqs = defaultdict(int)

for f, w in word_freqs:
    for l in w:
        letter_freqs[l] += f

write_freqs(letter_freqs, 'freq-letter.txt')

digram_freqs = defaultdict(int)

for f, w in word_freqs:
    w = ' ' + w + ' '
    for l0, l1 in zip(w, w[1:]):
        digram_freqs[l0 + l1] += f

write_freqs(digram_freqs, 'freq-digram.txt')

trigram_freqs = defaultdict(int)

for f, w in word_freqs:
    w = '  ' + w + '  '
    for l0, l1, l2 in zip(w, w[1:], w[2:]):
        if l0 == l1:
            l1 = '.'
        if l1 == l2:
            l2 = '.'
        trigram_freqs[l0 + l1 + l2] += f

write_freqs(trigram_freqs, 'freq-trigram.txt')
