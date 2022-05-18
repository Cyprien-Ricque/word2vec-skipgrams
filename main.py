import warnings
import json

warnings.filterwarnings("ignore", category=UserWarning)

from src.model import SkipGramNegativeSampling
from src.config import Config
from src.corpus import Corpus
from src.nstable import NegativeSamplingTable
from src.context_manager import ContextManager

import torch


if __name__ == '__main__':
    warnings.filterwarnings("default", category=UserWarning)

    c = Corpus(filename=Config.get('corpus-name'), force_recreate=False)

    Config.set('corpus', c.wiki)

    if Config.get('device') == 'cuda' and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print('Using GPU.')
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        print('Using CPU.')

    ns_table = NegativeSamplingTable(
        corpus=Config.get('corpus'),
        table_size=Config.get('ns-table-size'),
        unknown_str=Config.get('unknown-str'),
        filename=Config.get('ns-table-file'),
        max_voc_size=Config.get('voc-size'),
        force_create=True
    )

    model = SkipGramNegativeSampling(
        token2id=Config.get('corpus').dictionary.token2id,
        embedding_dim=Config.get('embedding-dim')
    )

    trainer = ContextManager(
        corpus=Config.get('corpus'),
        model=model,
        ns_table=ns_table.table,
        batch_size=Config.get('batch-size'),
        negative_samples=Config.get('negative-samples'),
        epochs=Config.get('epochs'),
        lr=Config.get('lr'),
        token2id=Config.get('corpus').dictionary.token2id,
        ctx_window_size=Config.get('context-size')
    )

    trainer.fit()

    print('Save model')
    torch.save(model, f'./models/{Config.get("model-file")}')

    print('Evaluate model')
    r = model.evaluateWordEmbeddings()

    # Save results
    r['config'] = Config.read()
    results_str = f'{json.dumps(r)},\n'
    with open('results.json', 'a') as f:
        f.write(results_str)

    print(results_str)
