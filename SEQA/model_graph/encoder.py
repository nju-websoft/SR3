from pytorch_transformers import BertModel
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from pytorch_transformers import BertModel
import torch.nn as nn
import torch
class Bert(nn.Module):
    def __init__(self, model_name, temp_dir, finetune=True):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained(model_name, cache_dir=temp_dir)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec
class RelationEncoder(nn.Module):
    def __init__(self, vocab, rel_dim, embed_dim, hidden_size, num_layers, dropout, bidirectional=True):
        super(RelationEncoder, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rel_embed = nn.Embedding(vocab.size, rel_dim)
        self.rnn = nn.GRU(
            input_size=rel_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout if num_layers > 1 else 0.,
            bidirectional=bidirectional
        )
        tot_dim = 2 * hidden_size if bidirectional else hidden_size
        self.out_proj = nn.Linear(tot_dim, embed_dim)
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)
        self.rel_embed.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, src_tokens, src_lengths):
        seq_len, bsz = src_tokens.size()
        ###
        sorted_src_lengths, indices = torch.sort(src_lengths, descending=True)
        sorted_src_tokens = src_tokens.index_select(1, indices)
        ###
        x = self.rel_embed(sorted_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, sorted_src_lengths.data.tolist())

        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        _, final_h = self.rnn(packed_x, h0)

        if self.bidirectional:
            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz,
                                                                                                -1)

            final_h = combine_bidir(final_h)

        ###
        _, positions = torch.sort(indices)
        final_h = final_h.index_select(1, positions)  # num_layers x bsz x hidden_size

        output = self.out_proj(final_h[-1])

        return output
