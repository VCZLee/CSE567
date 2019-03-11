import nltk
import numpy as np
from scipy.stats.mstats import gmean

f = open("news.2007.de.shuffled", "r")
if f.mode == 'r':
    contents = f.read()

tokens = nltk.word_tokenize(contents, "german")
tokens = [token.lower() for token in tokens]


tokens = nltk.probability.FreqDist(tokens)
word = "arbeitsmarktpolitik"

substrings = set([])
for i in range(len(word)):
    for j in range(len(word) - i):
        substrings.add(word[j:j+i+1])


for substring in list(substrings):
    if (len(substring) < 3) or (substring not in tokens): # enforce minimum token size of 3
        substrings.remove(substring)



validSubstrings = sorted(list(substrings), key=len)

  # below code was retrieved from
# https://codereview.stackexchange.com/questions/197558/determine-all-ways-a-string-can-be-split-into-valid-words-given-a-dictionary-of
import collections.abc


class TrieNode(collections.abc.MutableMapping):
    def __init__(self, k):
        self._data = {}
        self._value = k
        self.end = False

    @staticmethod
    def build(iterable):
        root = TrieNode(None)
        for key in iterable:
            root[key] = True
        return root

    @property
    def value(self):
        return self._value

    def _keys(self, key):
        partial = ''
        for k in key:
            partial += k
            yield k, partial

    def _walk(self, data, key, *, build=False):
        if not key:
            raise ValueError()

        node = data
        if not build:
            for k in key[:-1]:
                node = node._data[k]
        else:
            for k, key_ in self._keys(key[:-1]):
                node = node._data.setdefault(k, TrieNode(key_))
        return key[-1], node

    def __getitem__(self, key):
        key, node = self._walk(self, key)
        return node._data[key]

    def __setitem__(self, key, value):
        k, node = self._walk(self, key, build=True)
        node = node._data.setdefault(k, TrieNode(key))
        node.end = value

    def __delitem__(self, key):
        key, node = self._walk(self, key)
        del node._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def word_break(word_string, words):
    words = TrieNode.build(words)
    output = {0: [[]]}
    partials = []
    for i, k in enumerate(word_string, 1):
        new_partials = []
        for partial in partials + [words]:
            partial = partial.get(k)
            if partial is None:
                continue

            new_partials.append(partial)
            if not partial.end:
                continue

            val = partial.value
            prevs = output.get(i - len(val))
            if prevs is not None:
                output.setdefault(i, []).extend([p + [val] for p in prevs])
        partials = new_partials
    return output[len(word_string)]


decompositions = word_break(word, validSubstrings)
print(decompositions)

counts = [np.array([tokens[token] for token in decomposition]) for decomposition in decompositions]
print(counts)
counts = [gmean(frequency) for frequency in counts]
print(counts)
print(decompositions[np.argmax(counts)])
