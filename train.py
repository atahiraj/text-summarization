
import torch

from torchtext import data, vocab

import os
import argparse

from fastai.text import *

DATA_PATH = os.path.join("./", "data/")
SAMPLE_DATA_PATH = os.path.join(DATA_PATH, "sample_data/")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed_data/")

class Train(object):
    def __init__(self, opts):
        pass

class ModelData():
    """Encapsulates DataLoaders and Datasets for training, validation, test. Base class for fastai *Data classes."""
    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path,self.trn_dl,self.val_dl,self.test_dl = path,trn_dl,val_dl,test_dl

    @classmethod
    def from_dls(cls, path,trn_dl,val_dl,test_dl=None):
        #trn_dl,val_dl = DataLoader(trn_dl),DataLoader(val_dl)
        #if test_dl: test_dl = DataLoader(test_dl)
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg
    @property
    def is_multi(self): return self.trn_ds.is_multi
    @property
    def trn_ds(self): return self.trn_dl.dataset
    @property
    def val_ds(self): return self.val_dl.dataset
    @property
    def test_ds(self): return self.test_dl.dataset
    @property
    def trn_y(self): return self.trn_ds.y
    @property
    def val_y(self): return self.val_ds.y

class BatchTuple():
    def __init__(self, dataset, x_var, y_var):
        self.dataset, self.x_var, self.y_var = dataset, x_var, y_var
        
    def __iter__(self):
        for batch in self.dataset:
            x = getattr(batch, self.x_var) 
            y = getattr(batch, self.y_var)                 
            yield (x, y)
            
    def __len__(self):
        return len(self.dataset)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument(
        "--gpu", action="store_true", help="Use gpu",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.path.join("./", "data/processed_data/train.csv"),
        help="Path to training set"
    )
    parser.add_argument(
        "--valid",
        type=str,
        default=os.path.join("./", "data/processed_data/valid.csv"),
        help="Path to validation set"
    )

    opts = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and opts.gpu else "cpu")
    
    #train = Train(opts)


    tokenizer = data.get_tokenizer("spacy")
    TEXT = data.Field(tokenize=tokenizer, lower=True, eos_token='_eos_')



    trn_data_fields = [("source", TEXT),
                    ("target", TEXT)]

    trn, vld = data.TabularDataset.splits(path="data/processed_data/",
                                        train='train_sample.csv', validation='valid_sample.csv',
                                        format='csv', skip_header=True, fields=trn_data_fields)

    pre_trained_vector_type = 'glove.6B.200d' 
    TEXT.build_vocab(trn, vectors=pre_trained_vector_type)

    print(TEXT.vocab.freqs.most_common(10))

    batch_size = 64

    train_iter, val_iter = data.BucketIterator.splits(
                            (trn, vld), batch_sizes=(batch_size, int(batch_size*1.6)),
                            device="cpu", 
                            sort_key=lambda x: len(x.source),
                            shuffle=True, sort_within_batch=False, repeat=False)


    train_iter_tuple = BatchTuple(train_iter, "source", "target")
    val_iter_tuple = BatchTuple(val_iter, "source", "target")

    print(next(iter(train_iter_tuple)))

    model_data = ModelData(DATA_PATH, trn_dl=train_iter_tuple, val_dl=val_iter_tuple)


    print(len(model_data.trn_dl), len(model_data.val_dl), len(TEXT.vocab))

    t, z = next(model_data.trn_dl.__iter__())
    print(t.size(), z.size())

    sample_source = t.transpose(1,0)[0].data.cpu().numpy()
    sample_target = z.transpose(1,0)[0].data.cpu().numpy()

    print("source:\n%s \n\ncorresponding tensor:\n%s \n" %(' '.join([TEXT.vocab.itos[o] for o in sample_source]), sample_source))
    print("target:\n%s \n\ncorresponding tensor:\n%s \n" %(' '.join([TEXT.vocab.itos[o] for o in sample_target]), sample_target))
    