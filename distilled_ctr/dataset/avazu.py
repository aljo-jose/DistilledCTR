import shutil
import struct
from collections import defaultdict
from pathlib import Path
import joblib

import numpy as np
import torch.utils.data
from tqdm import tqdm


class AvazuDataset(torch.utils.data.Dataset):
    """
    Avazu Click-Through Rate Prediction Dataset

    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature

    :param dataset_path: avazu train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold

    Reference
        https://www.kaggle.com/c/avazu-ctr-prediction
    """

    def __init__(self, dataset_path=None, cache_path='.avazu', rebuild_cache=False, min_threshold=4):
        self.NUM_FEATS = 22
        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            cache, field_dims = self.__build_cache(dataset_path, cache_path)
            self.save_cache(cache_path, cache, field_dims)
        else:
            cache, field_dims = self.load_cache(cache_path)
        self.cache, self.field_dims = cache, field_dims

    def save_cache(self, cache_path, cache, field_dims):
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, cache_path+'/cache.gz')
        joblib.dump(field_dims, cache_path+'/field_dims.gz')
    
    def load_cache(self, cache_path):
        cache = joblib.load(cache_path+'/cache.gz')
        field_dims = joblib.load(cache_path+'/field_dims.gz')
        return cache, field_dims

    def __getitem__(self, index):
        record = torch.tensor(self.cache[index], dtype=torch.long)
        return record[1:], record[0]

    def __len__(self):
        return len(self.cache)

    def __build_cache(self, path, cache_path):
        feat_mapper, defaults = self.__get_feat_mapper(path)
        
        field_dims = np.zeros(self.NUM_FEATS, dtype=np.int32)
        for i, fm in feat_mapper.items():
            field_dims[i - 1] = len(fm) + 1
        cache = self.__yield_buffer(path, feat_mapper, defaults)
        return cache, field_dims

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i + 1]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: preprocessing')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.int32)
                np_array[0] = int(values[1])
                for i in range(1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i+1], defaults[i])
                #buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                buffer.append(np_array)
                item_idx += 1
                # if item_idx % buffer_size == 0:
                #     yield buffer
                #     buffer.clear()
            #yield buffer
            return np.array(buffer)

if __name__ == "__main__":
    ds = AvazuDataset(
        dataset_path='data/avazu/train', 
        cache_path='.avazu', 
        rebuild_cache=True, 
        min_threshold=10)
    n = ds.__len__()
    x = ds.__getitem__(index=0)