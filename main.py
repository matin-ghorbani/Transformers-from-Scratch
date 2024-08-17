import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * \
               heads == self.embed_size, 'Embed size needs to be divisible by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc = nn.Linear(
            self.heads * self.head_dim, self.embed_size
        )

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        num_training_examples = query.shape[0]
        value_len, keys_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(num_training_examples,
                                value_len, self.heads, self.head_dim)
        keys = keys.reshape(num_training_examples, keys_len,
                            self.heads, self.head_dim)
        queries = query.reshape(num_training_examples,
                                query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Queries shape: (N, query_len, heads, heads_dim)
        #    keys shape: (N, key_len, heads, heads_dim)
        #  Energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum(
            'nqhd,nkhd->nhqk', (queries, keys)
        )

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(
            energy / (self.embed_size ** (1 / 2)), dim=3
        )

        #    Attention shape: (N, heads, query_len, key_len)
        #       Values shape: (N, value_len, heads, heads_dim)
        # After einsum shape: (N, query_len, heads, heads_dim) then flatten last two dimensions
        out = torch.einsum('nhql,nlhd->nqhd', (attention, values)).reshape(
            num_training_examples, query_len, self.heads * self.head_dim
        )


        return self.fc(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention = self.attention(value, key, query, mask)

        x = self.dropout(
            self.norm1(attention + query)
        )
        forward = self.feed_forward(x)
        out = self.dropout(
            self.norm2(attention + x)
        )

        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout=dropout,
                             forward_expansion=forward_expansion)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        N, sequence_len = x.shape

        positions = torch.arange(0, sequence_len).expand(
            N, sequence_len).to(self.device)
        out = self.dropout(
            self.word_embedding(x) + self.positional_embedding(positions)
        )

        for layer in self.layers:
            # value, key and query all going to be the same
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            value: torch.Tensor,
            key: torch.Tensor,
            src_mask: torch.Tensor,
            target_mask: torch.Tensor
    ) -> torch.Tensor:
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(
            self.norm(attention + x)
        )
        out = self.transformer_block(value, key, query, src_mask)

        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_len,
            device,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion,
                         dropout=dropout, device=device)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            encoder_out: torch.Tensor,
            src_mask: torch.Tensor,
            target_mask: torch.Tensor
    ) -> torch.Tensor:
        N, sequence_len = x.shape

        positions = torch.arange(0, sequence_len).expand(N, sequence_len).to(self.device)
        x = self.dropout((
                self.word_embedding(x) + self.position_embedding(positions)
        ))

        layer: DecoderBlock
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, target_mask)

        return self.fc(x)


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            target_vocab_size,
            src_pad_idx,
            target_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            max_len=100
    ) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            self.device,
            forward_expansion,
            dropout,
            max_len
        )
        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_len,
            self.device,
        )

        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        src_mask: torch.Tensor = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)

        return src_mask.to(self.device)

    def make_target_mask(self, target: torch.Tensor) -> torch.Tensor:
        n, target_len = target.shape
        target_mask = torch.tril(
            torch.ones((target_len, target_len))
        ).expand(
            n, 1, target_len, target_len
        )

        return target_mask.to(self.device)

    def forward(self, src: torch.Tensor, target: torch.Tensor):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)

        encode_src = self.encoder(src, src_mask)
        return self.decoder(target, encode_src, src_mask, target_mask)


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.tensor([
        [1, 5, 6, 4, 3, 9, 5, 2, 0],
        [1, 8, 7, 3, 4, 5, 6, 7, 2]
    ]).to(device)
    target = torch.tensor([
        [1, 7, 4, 3, 5, 9, 2, 0],
        [1, 5, 6, 2, 4, 7, 6, 2]
    ]).to(device)

    src_pad_idx = 0
    src_vocab_size = 10

    target_pad_idx = 0
    target_vocab_size = 10

    transformer = Transformer(src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx).to(device)
    out = transformer(x, target[:, :-1])
    print(f'{out.shape = }')


if __name__ == '__main__':
    main()
