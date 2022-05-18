from tqdm import tqdm
import os


class NegativeSamplingTable:
    def __init__(self, corpus, max_voc_size, table_size, unknown_str, filename, exp=0.75, force_create=False):
        self.corpus = corpus
        self.max_voc_size = max_voc_size
        self.table_size = table_size
        self.low_freq_str = unknown_str
        self.exp = exp
        self.filename = filename

        if os.path.exists(self.filename) and not force_create:
            self.table = self.load()
        else:
            self.create()
            self.save()

    def create(self):
        frequencies = self.corpus.dictionary.cfs

        frequencies = sorted(frequencies.items(), key=lambda p: (p[1], p[0]), reverse=True)

        sum_frequencies = 1
        if len(frequencies) > self.max_voc_size - 1:
            sum_frequencies = sum(f for _, f in frequencies[self.max_voc_size - 1:])

        frequencies = [(self.low_freq_str, sum_frequencies)] + frequencies[:self.max_voc_size - 1]

        table = {}
        sum_frequencies = 0
        for w, freq in frequencies:
            ns_freq = freq ** self.exp
            table[w] = ns_freq
            sum_frequencies += ns_freq

        scaler = self.table_size / sum_frequencies

        self.table = [(w, freq, int(round(table[w] * scaler))) for w, freq in frequencies]
        print('NegativeSamplingTable', self.table[:3])

    def load(self):
        with open(self.filename) as file:
            out = []
            for line in file:
                t = line.split()
                out.append((t[0], int(t[1]), int(t[2])))
            return out

    def save(self):
        with open(self.filename, 'w', encoding='utf-16') as f:
            for w, fr, ns in self.table:
                print(f'{w} {fr} {ns}', file=f)


