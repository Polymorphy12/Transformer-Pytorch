from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        # ext : A tuple containing the extension to path for each language.
        self.ext = ext
        # tokenize_en, tokenize_de는 각각 영어와 독일어 텍스트를 토큰화시키는 함수다.
        # 그냥 변수가 아니다.
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        # <sos>
        self.init_token = init_token
        # <eos>
        self.eos_token = eos_token
        print('start dataset initializing')
    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True,
                                batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True,
                                batch_first=True)
        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True,
                                batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token,
                                eos_token=self.eos_token, lower=True,
                                batch_first=True)
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext,
                                                            fields=(self.source, self.target))
        return train_data, valid_data, test_data
    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)
    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('Dataset initialized.')
        return train_iterator, valid_iterator, test_iterator