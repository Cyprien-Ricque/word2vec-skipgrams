from tqdm import tqdm
import time

from src.model import SkipGramNegativeSampling, nn
import torch


class ContextManager:

    def __init__(self, corpus, model, ns_table, batch_size, negative_samples, epochs, lr, token2id, ctx_window_size):
        self.corpus = corpus
        self.model: SkipGramNegativeSampling = model
        self.negative_samples = negative_samples
        self.batch_size = batch_size
        self.token2id = token2id
        self.ctx_window_size = ctx_window_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.epochs = epochs

        # Build the negative sampling table.
        ns_table_expanded = []
        for i, (_, _, count) in enumerate(ns_table):
            ns_table_expanded.extend([i] * count)
        self.ns_table = torch.as_tensor(ns_table_expanded)

        y_pos = torch.ones((batch_size, 1))
        y_neg = torch.zeros((batch_size, self.negative_samples))
        self.y = torch.cat([y_pos, y_neg], dim=1)

        print('Estimating skip grams count...')
        text_size = 66252901
        # text_size = sum(len(i) for i in corpus.get_texts())
        print(f'{text_size=}')
        self.batches_count_estimation = ((text_size * ctx_window_size * 2) // self.batch_size) + 1
        # self.batches_count_estimation = 3033
        print('batches_count_estimation', self.batches_count_estimation)

    def negativeSamples(self, batch_size):
        neg_sample_ixs = torch.randint(len(self.ns_table), (batch_size, self.negative_samples))
        return self.ns_table.take(neg_sample_ixs)

    def fit(self):
        epoch_loss = -1

        for epoch in range(1, self.epochs + 1):
            sum_loss = 0

            progress_bar = tqdm(enumerate(self.batches()), total=self.batches_count_estimation)

            for batch_id, (t, c_pos, c_neg) in progress_bar:
                progress_bar.set_description(f'Epoch {epoch}. Loss: {epoch_loss:.4f}')

                batch_size = len(t)

                t = torch.as_tensor(t)
                c_pos = torch.as_tensor(c_pos).view(batch_size, 1)
                c = torch.cat([c_pos, c_neg], dim=1)

                self.optimizer.zero_grad()
                y_hat = self.model(t, c)

                loss = self.loss(y_hat, self.y[:batch_size])
                loss.backward()
                self.optimizer.step()

                sum_loss += loss.item()
                epoch_loss = round(sum_loss / (batch_id + 1), 4)

    def batches(self):
        targets, contexts = [], []

        for tokens in self.corpus.get_texts():
            encoded = [self.token2id[t] for t in tokens]

            for i_t, target in enumerate(encoded):
                start = max(0, i_t - self.ctx_window_size)
                end = min(i_t + self.ctx_window_size + 1, len(encoded))

                for context_i in range(start, end):
                    if context_i == i_t:
                        continue

                    targets.append(target)
                    contexts.append(encoded[context_i])

                    if len(targets) >= self.batch_size:
                        yield targets, contexts, self.negativeSamples(len(targets))
                        targets, contexts = [], []

        if len(targets) > 0:
            yield targets, contexts, self.negativeSamples(len(targets))



