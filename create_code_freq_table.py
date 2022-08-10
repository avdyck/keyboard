import os
import string

path = 'C:\\Users\\AlexanderVanDyck\\IdeaProjects\\conundra-optitour'

accepted_letters = set(string.ascii_lowercase)

bigrams = {}
trigrams = {}

javafiles = []

for root, subdirs, files in os.walk(path):
    for file in files:
        if file.endswith('.java'):
            javafiles.append(os.path.join(root, file))

for _ in range(len(javafiles)):
    print('.', end='')
print()

for i, file in enumerate(javafiles):
    print('.', end='')
    with open(file, 'r', encoding='utf-8') as fh:
        for line in fh:
            read = line.lstrip().lower()
            if read.startswith('package') or read.startswith('import'):
                continue

            for a, b in zip(read, read[1:]):
                bi = a + b
                if not all(x in accepted_letters for x in bi):
                    continue
                if bi not in bigrams:
                    bigrams[bi] = 1
                else:
                    bigrams[bi] += 1

            for a, b, c in zip(read, read[1:], read[2:]):
                tri = a + b + c
                if not all(x in accepted_letters for x in tri):
                    continue
                if tri not in trigrams:
                    trigrams[tri] = 1
                else:
                    trigrams[tri] += 1
print()

bigrams_list = [(f, w) for w, f in bigrams.items()]
bigrams_list.sort(reverse=True)

trigrams_list = [(f, w) for w, f in trigrams.items()]
trigrams_list.sort(reverse=True)

print(bigrams_list)
print(trigrams_list)

with open('code-bi.txt', 'w') as fh:
    for f, bi in bigrams_list:
        fh.write(f'{bi}\t{f}\n')

with open('code-tri.txt', 'w') as fh:
    for f, tri in trigrams_list:
        fh.write(f'{tri}\t{f}\n')
