import os
import string
from collections import defaultdict

path = 'C:\\Users\\AlexanderVanDyck\\IdeaProjects\\conundra-optitour'

accepted_letters = set(string.ascii_lowercase)

bigrams = {}
trigrams = {}

javafiles = []

english_most_frequent = 0
with open('english.txt', 'r') as fh:
    for line in fh:
        _, f = line.strip().split('\t')
        english_most_frequent = max(english_most_frequent, int(f))

for root, subdirs, files in os.walk(path):
    for file in files:
        if file.endswith('.java'):
            javafiles.append(os.path.join(root, file))

for _ in range(len(javafiles)):
    print('.', end='')
print()

words = defaultdict(int)
for i, file in enumerate(javafiles):
    print('.', end='')
    with open(file, 'r', encoding='utf-8') as fh:
        for line in fh:
            read = line.lstrip().lower()
            if read.startswith('package') or read.startswith('import'):
                continue

            read = ''.join(' ' if l not in accepted_letters else l for l in read)
            for w in read.split(' '):
                if w:
                    words[w] += 1
print()

result = [(f, w) for w, f in words.items()]
result.sort(reverse=True)
most_frequent = result[0][0]

with open('code.txt', 'w') as fh:
    for f, w in result:
        fh.write(f'{w}\t{(f * english_most_frequent / most_frequent)}\n')
