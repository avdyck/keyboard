import string

import unidecode as unidecode

result = []
accepted_letters = set(string.ascii_lowercase + '#\'\"')

english_most_frequent = 0
with open('english.txt', 'r') as fh:
    for line in fh:
        _, f = line.strip().split('\t')
        english_most_frequent = max(english_most_frequent, int(f))

dutch_most_frequent = 0
with open('vocab', 'r', encoding='utf-8') as fh:
    for line in fh:
        try:
            w, f = line.strip().split()
            f: int = int(f)
            w: str = unidecode.unidecode(w).lower()
            if f >= 1_000 and len(w) > 1:
                if not all(x in accepted_letters for x in w):
                    continue

                w = ''.join(x for x in w if x in string.ascii_lowercase)
                if len(w) > 1:
                    dutch_most_frequent = max(dutch_most_frequent, f)
                    result.append((f, w))
        except Exception:
            pass

result.sort(reverse=True)
result = [
    (f * english_most_frequent / dutch_most_frequent, w)
    for f, w in result
]

with open('dutch.txt', 'w') as fh:
    for f, w in result:
        fh.write(f'{w}\t{f}\n')
