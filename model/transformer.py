import copy

import torch
import torch.nn as nn

from torch.autograd import Variable

import math


'''
Reference
1. https://cpm0722.github.io/pytorch-implementation/transformer
2. https://github.com/hyunwoongko/transformer
3. http://nlp.seas.harvard.edu/2018/04/03/attention.html
'''
class ScaledDotattention(nn.Module):
    def __init__(self):
        super(ScaledDotattention, self).__init__()

    def forward(self, q, k, v, mask):
        # shape : [n_batch, seq_len, d_k]
        query = q
        key = k
        value = v

        # shape : [n_batch, seq_len, seq_len]
        score = torch.matmul(query, key.transpose(-2, -1))
        score = score / math.sqrt(key.size(-1))
        if mask is not None:
            # AutoRegressive 성질을 만족하기 위한 masking
            # mask가 0인 부분을 -inf로 채운다.
            score = score.masked_fill(mask == 0, -1e9)
        # shape : [n_batch, seq_len, seq_len]
        prob = nn.Softmax(dim=-1)(score)
        # shape : [n_batch, seq_len, d_k]
        out = torch.matmul(prob, value)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_embed, h, device):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_embed = d_embed
        self.h = h

        self.Q = nn.Linear(d_embed, d_model).to(device)
        self.K = nn.Linear(d_embed, d_model).to(device)
        self.V = nn.Linear(d_embed, d_model).to(device)

        self.self_attention = ScaledDotattention()

        self.out_fc = nn.Linear(d_model, d_embed).to(device)

    def forward(self, q, k, v, mask=None):
        # shape : [n_batch, seq_len, d_embed] -> [n_batch, seq_len, d_model)
        query = self.Q(q)
        key = self.K(k)
        value = self.V(v)

        n_batch = query.shape[0]

        # shape : [n_batch, self.h, seq_len, d_k]
        query = self.split(query)
        key = self.split(key)
        value = self.split(value)

        # shape : [n_batch, self.h, seq_len, d_k]
        out = self.self_attention(query, key, value, mask)
        # shape : [n_batch, seq_len, self.h, d_k]
        out = out.transpose(1, 2)
        # shape : [n_batch, seq_len, d_model]
        out = out.contiguous().view(n_batch, -1, self.d_model)
        # shape : [n_batch, seq_len, d_embed]
        out = self.out_fc(out)
        return out

    def split(self, tensor):
        # tensor_shape : [n_batch, seq_len, d_model]
        n_batch, seq_len, d_model = tensor.shape

        # d_model 크기를 h로 쪼갠다.
        # shape : [n_batch, seq_len, h, d_k]
        out = tensor.view(n_batch, seq_len, self.h, d_model // self.h)
        # shape : [n_batch, h, seq_len, d_k]
        # self attention을 계산할 때 맨 마지막 [seq_len, d_k]를 기준으로
        # matrix 연산을 하기 때문에 이렇게 transpose 시켜준다.
        out = out.transpose(1, 2)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, device, d_model=512, d_ff=2048, drop_prob=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff).to(device)
        self.linear2 = nn.Linear(d_ff, d_model).to(device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob).to(device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


'''
LayerNorm에 대한 자세한 내용은 다음을 참고할 것.
https://arxiv.org/pdf/1607.06450.pdf

nn.Parameter : 이를 취해주면 텐서를 학습가능한 파라미터로 만들 수 있다.
'''


class LayerNorm(nn.Module):
    def __init__(self, d_model, device, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)).to(device)
        self.beta = nn.Parameter(torch.zeros(d_model)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


'''
Transformer의 인풋으로 사용할 임베딩을 정의하는 클래스이다.
'''


class TransformerEmbedding(nn.Module):
    def __init__(self, d_embed, vocab, device, max_seq_len=5000):
        super(TransformerEmbedding, self).__init__()
        # self.embedding = nn.Sequential(
        #     Embedding(d_embed=d_embed, vocab=vocab, device=device),
        #     PositionalEncoding(d_embed=d_embed, device=device)
        # )
        self.emb = Embedding(d_embed=d_embed, vocab=vocab, device=device)
        self.pos_emb = PositionalEncoding(d_embed=d_embed, device=device)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        emb = self.emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(emb + pos_emb)


'''
nn.Embedding(num_embeddings, embedding_dim)에 대해:
num_embeddings : 임베딩을 할 단어들의 개수, 다시 말해 단어 집합을 말한다.
embedding_dim : 임베딩할 벡터의 차원이다. 사용자가 정하는 하이퍼파라미터이다.
https://wikidocs.net/64779

* Transformer에서는 embedding도 scaling을 해준다.
'''


class Embedding(nn.Module):
    def __init__(self, d_embed, vocab, device):
        super(Embedding, self).__init__()
        self.vocab = vocab
        self.d_embed = d_embed
        self.device = device
        self.embedding = nn.Embedding(len(vocab), d_embed).to(self.device)

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_embed, device, max_len=5000):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_embed, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_embed, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_embed)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_embed)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_embed, d_ff, h, device, drop_prob=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, d_embed=d_embed, h=h, device=device)
        self.norm1 = LayerNorm(d_model=d_model, device=device)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop_prob=drop_prob, device=device)
        self.norm2 = LayerNorm(d_model=d_model, device=device)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        # residual connection을 하기위한 변수
        _x = x
        # self attention 계산
        x = self.multi_head_attention(q=x, k=x, v=x, mask=mask)

        # add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # position wise feed forward network
        _x = x
        x = self.feed_forward(x)

        # add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, d_embed, d_ff, h, vocab, device, drop_prob=0.1):
        super(Encoder, self).__init__()

        self.emb = TransformerEmbedding(d_embed=d_embed, vocab=vocab, device=device)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_embed=d_embed, d_ff=d_ff, h=h, drop_prob=drop_prob, device=device)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_embed, d_ff, h, device, drop_prob=0.1):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, d_embed=d_embed, h=h, device=device)
        self.norm1 = LayerNorm(d_model=d_model, device=device)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model, d_embed, h, device=device)
        self.norm2 = LayerNorm(d_model=d_model, device=device)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop_prob=drop_prob, device=device)
        self.norm3 = LayerNorm(d_model=d_model, device=device)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask, encoder_output, encoder_mask):
        # self attention을 계산한다.
        _x = x
        x = self.multi_head_attention(q=x, k=x, v=x, mask=mask)

        # add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if encoder_output is not None:
            # encoder_output을 받아들이는 self attention을 계산한다.
            _x = x
            x = self.enc_dec_attention(q=x, k=encoder_output, v=encoder_output, mask=encoder_mask)

            # add and norm
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        # Feed Forward Network
        _x = x
        x = self.feed_forward(x)

        # add and norm
        x = self.norm3(x + _x)
        x = self.dropout3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, d_embed, d_ff, h, vocab, device, drop_prob=0.1):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(d_embed=d_embed, vocab=vocab, device=device)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_embed=d_embed, d_ff=d_ff, h=h, drop_prob=drop_prob, device=device)
            for _ in range(num_layers)
        ])

        self.linear = nn.Linear(d_model, len(vocab)).to(device)

    def forward(self, x, mask, encoder_output, encoder_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask, encoder_output, encoder_mask)

        output = self.linear(x)
        return output


'''
<pad> token : 문장의 padding을 나타내는 토큰.
<sos> token : Start-of-string(sequence) 토큰.
<eos> token : End-of-sequence 토큰.
'''


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_vocab, dec_vocab,
                 num_layers, d_model, d_embed, d_ff, h, device, drop_prob=0.1):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

        self.device = device

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               d_embed=d_embed, d_ff=d_ff,
                               h=h, vocab=enc_vocab, drop_prob=drop_prob, device=device)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               d_embed=d_embed, d_ff=d_ff,
                               h=h, vocab=dec_vocab, drop_prob=drop_prob, device=device)

    def forward(self, src, trg):
        source_mask = self.make_pad_mask(src, src)
        source_target_mask = self.make_pad_mask(trg, src)
        target_mask = self.make_pad_mask(trg, trg).to(self.device) * self.make_no_peek_mask(trg, trg).to(self.device)

        encoder_output = self.encoder(src, source_mask)
        output = self.decoder(trg, target_mask, encoder_output, source_target_mask)
        return output

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # shape : [n_batch, 1, 1, len_k]
        # padding_index가 아닌것에 True, 맞는 것에 False를 매기는 mask
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # shape : [n_batch, 1, len_q, len_k]
        k = k.repeat(1, 1, len_q, 1)

        # shape : [n_batch, 1, len_q, 1]
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # shape : [n_batch, 1, len_q, len_k]
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    # target이 다음 단어를 보지 못하도록 하는 mask
    def make_no_peek_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor)

        return mask


