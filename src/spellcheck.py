import re
from collections import Counter

def words(text): 
    return re.findall(r'\w+', text.lower())

def P(word, WORDS): 
    "Probability of `word`."
    N=sum(WORDS.values())
    return WORDS[word] / N

def corrections(word, WORDS): 
    "Most probable spelling correction for word."
    return sorted(candidates(word, WORDS), key=lambda x: P(x, WORDS), reverse=True)[:5]

def candidates(word, WORDS): 
    "Generate possible spelling corrections for word."
    return (known([word], WORDS) or known(edits1(word), WORDS) or known(edits2(word), WORDS) or [word])

def known(words, WORDS): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz._- '
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correct_query(text, WORDS):
    corrected = []

    if text in WORDS:
        corrected.append(text)
    else:
        corrected.extend(corrections(text, WORDS))
    return corrected

def find_close(query, nodes):
    names, tags = nodes['name1'], nodes['email1'].apply(lambda x: x.split('@')[0])
    name_keys = correct_query(query, Counter(names))
    tag_keys = correct_query(query, Counter(tags))
    return list(sorted(set(nodes[(names.isin(name_keys))|(tags.isin(tag_keys))].index)))