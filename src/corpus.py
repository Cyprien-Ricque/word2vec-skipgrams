import os
from gensim.test.utils import datapath, get_tmpfile
from gensim import utils
from gensim.corpora import WikiCorpus, MmCorpus


class Corpus:
    def __init__(self, filename, force_recreate=False):
        self.folder = './data/'
        self.path_to_wiki_dump = self.folder + filename
        self.corpus_path = self.folder + filename + '_wikidump.txt'
        self.wiki: WikiCorpus = None

        self.create(force_recreate=force_recreate)

    def create(self, save=True, force_recreate=False):
        if not os.path.exists(self.corpus_path) or force_recreate:
            print('Load from origin', self.path_to_wiki_dump)
            self.wiki = WikiCorpus(fname=self.path_to_wiki_dump, processes=5)
            print('Loaded. length:', self.wiki.length)
            if save:
                print('Save corpus in', self.corpus_path)
                self.wiki.save(self.corpus_path)
        else:
            print('Load from saved', self.corpus_path)
            self.wiki = WikiCorpus.load(self.corpus_path)
