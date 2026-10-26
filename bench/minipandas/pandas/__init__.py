# minipandas: minimal pandas stand-in ONLY so that `vllm bench serve --dataset-name custom` can read a .jsonl
# (the vLLM container does not ship pandas and the server had no PyPI access). suite.sh mounts it only for T15 with real text.
import json

__version__ = "0.0-iggybench"


class _Row(dict):
    def to_dict(self):
        return dict(self)


class DataFrame:
    def __init__(self, records):
        self._f = [_Row(r) for r in records]
        self.columns = list(dict.fromkeys(k for r in self._f for k in r))

    def iterrows(self):
        return enumerate(self._f)

    def to_dict(self, orient="records"):
        return [dict(r) for r in self._f]

    def __len__(self):
        return len(self._f)

    def __getitem__(self, col):
        return [r.get(col) for r in self._f]


def read_json(path_or_buf=None, lines=False, **kw):
    with open(path_or_buf, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()] if lines else json.load(f)
    return DataFrame(records)
