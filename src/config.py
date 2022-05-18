import json
import logging
import sys
import json

files = [
    'enwiki-latest-pages-articles1.xml-p1p41242.bz2',
    'enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2'
]


class Config:
    def __init__(self):
        print("This class does not need instantiation", file=sys.stderr)

    __conf = {
        'corpus-name': files[0],
        'corpus': None,
        'device': 'cpu',
        'model-file': 'word_embd',

        'epochs': 1,
        'embedding-dim': 300,  # 200, 300
        'context-size': 1,
        'negative-samples': 3,
        'batch-size': 1 << 18,
        'lr': 1e-1,

        'ns-table-size': 1 << 24,

        'unknown-str': '<LOW_FREQ>',
        'ns-table-file': 'ns_table.txt',
        'voc-size': 1000000,

    }
    __setters = ['corpus', 'epochs', 'context-size', 'negative-sample', 'lr', 'model-file']
    __export = ['corpus-name', 'epochs', 'embedding-dim', 'context-size', 'negative-samples', 'batch-size', 'lr', 'model-file']

    @staticmethod
    def read():
        return {a: b for a, b in Config.__conf.items() if a in Config.__export}

    @staticmethod
    def config(name):
        return Config.__conf[name]

    @staticmethod
    def get(idx):
        return Config.config(idx)

    @staticmethod
    def set(name, value):
        if name in Config.__setters:
            Config.__conf[name] = value
            config_log.info(f'{name} becomes {value}')
        else:
            raise NameError("Name not accepted in set() method")


config_log = logging.getLogger('config')
config_log.setLevel(logging.INFO)
config_log.info(f'Config set to {Config.read()}')
Config.set('model-file', f'{Config.get("model-file")}_{Config.get("corpus-name")}_{Config.get("embedding-dim")}_{Config.get("context-size")}_{Config.get("batch-size")}_{Config.get("negative-samples")}_{Config.get("epochs")}')