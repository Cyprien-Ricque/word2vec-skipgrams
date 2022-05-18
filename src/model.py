import gensim
from gensim.test.utils import datapath
import torch
import torch.nn as nn

torch.manual_seed(123)


class SkipGramNegativeSampling(nn.Module):

    def __init__(self, token2id, embedding_dim):
        super().__init__()

        V = len(token2id)
        print('V', V, 'embedding_dim', embedding_dim)
        self.embd = nn.Embedding(V + 1, embedding_dim)
        self.token2id = token2id

        self.wv: gensim.models.keyedvectors.KeyedVectors = None

    def forward(self, target, context):
        """
            target: target word ids (1-dimensional tensor)
            context: positive and negative context for each target (2-dimensional tensor)
        """

        target_emb = self.embd(target)  # Embed target

        n_batch, emb_dim = target_emb.shape
        n_context = context.shape[1]

        target_emb = target_emb.view(n_batch, 1, emb_dim)  # Reshape for batch matmul
        context_emb = self.embd(context).transpose(1, 2)  # Embed context

        dots = target_emb.bmm(context_emb).view(n_batch, n_context)  # Dot product over *context:1target

        return dots

    def __createWV(self):
        if self.wv is not None:
            return
        self.wv = gensim.models.keyedvectors.KeyedVectors(vector_size=self.embd.weight.shape[1], count=self.embd.weight.shape[0])
        self.wv.vectors = self.embd.weight.cpu().detach().numpy()
        self.wv.key_to_index = self.token2id
        self.wv.index_to_key = list(self.token2id)

    def evaluateWordPairs(self):
        self.__createWV()

        (pearson,
         spearman,
         ratio_unknown_words) = self.wv.evaluate_word_pairs(datapath("wordsim353.tsv"))

        result = {
            'pearson correlation': pearson[0],
            'pearson p-value': pearson[1],
            'spearman correlation': spearman.correlation,
            'spearman p-value': spearman.pvalue,
            'ratio unknown words': ratio_unknown_words
        }

        return result

    def evaluateWordAnalogies(self):
        self.__createWV()
        result = {}

        try:
            score, _ = self.wv.evaluate_word_analogies(datapath('questions-words.txt'))
            result = {'score': score}
        except IndexError:
            print('Error while computing self.wv.evaluate_word_analogies(datapath(\'questions-words.txt\'))')

        return result

    def evaluateWordEmbeddings(self):
        return {
            'word pairs': self.evaluateWordPairs(),
            'word analogies': self.evaluateWordAnalogies()
        }

