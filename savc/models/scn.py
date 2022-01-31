import torch
from torch.nn.parameter import Parameter
from typing import Tuple
from constant import DEVICE


class SemanticLSTM(torch.nn.Module):
    """
    LSTM with semantic support
    Details: https://arxiv.org/abs/1909.00121
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        embed_size: int,
        semantic_size: int,
        cnn_feature_size: int,
        vocab_size: int,
        timestep: int = 80,
        drop_out_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.semantic_size = semantic_size
        self.cnn_feature_size = cnn_feature_size
        self.vocab_size = vocab_size
        self.timestep = timestep

        self._init_weights()
        self.caption_embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.linear_last = torch.nn.Linear(hidden_size, vocab_size)
        self.dropout_caption_embed = torch.nn.Dropout(drop_out_rate)
        self.dropout_cnn = torch.nn.Dropout(drop_out_rate)

    def _get_weight(self, size: Tuple):
        weight = torch.empty(size)
        weight = torch.nn.init.xavier_normal_(weight)
        return Parameter(weight)

    def _get_bias(self, size: Tuple):
        return Parameter(torch.zeros(size))

    def _init_weights(self):
        self.wa_i = self._get_weight((self.embed_size, self.input_size))
        self.wa_f = self._get_weight((self.embed_size, self.input_size))
        self.wa_o = self._get_weight((self.embed_size, self.input_size))
        self.wa_g = self._get_weight((self.embed_size, self.input_size))

        self.wb_i = self._get_weight((self.semantic_size, self.input_size))
        self.wb_f = self._get_weight((self.semantic_size, self.input_size))
        self.wb_o = self._get_weight((self.semantic_size, self.input_size))
        self.wb_g = self._get_weight((self.semantic_size, self.input_size))

        self.wc_i = self._get_weight((self.input_size, self.hidden_size))
        self.wc_f = self._get_weight((self.input_size, self.hidden_size))
        self.wc_o = self._get_weight((self.input_size, self.hidden_size))
        self.wc_g = self._get_weight((self.input_size, self.hidden_size))

        self.ua_i = self._get_weight((self.hidden_size, self.input_size))
        self.ua_f = self._get_weight((self.hidden_size, self.input_size))
        self.ua_o = self._get_weight((self.hidden_size, self.input_size))
        self.ua_g = self._get_weight((self.hidden_size, self.input_size))

        self.ub_i = self._get_weight((self.semantic_size, self.input_size))
        self.ub_f = self._get_weight((self.semantic_size, self.input_size))
        self.ub_o = self._get_weight((self.semantic_size, self.input_size))
        self.ub_g = self._get_weight((self.semantic_size, self.input_size))

        self.uc_i = self._get_weight((self.input_size, self.hidden_size))
        self.uc_f = self._get_weight((self.input_size, self.hidden_size))
        self.uc_o = self._get_weight((self.input_size, self.hidden_size))
        self.uc_g = self._get_weight((self.input_size, self.hidden_size))

        self.ca_i = self._get_weight((self.cnn_feature_size, self.input_size))
        self.ca_f = self._get_weight((self.cnn_feature_size, self.input_size))
        self.ca_o = self._get_weight((self.cnn_feature_size, self.input_size))
        self.ca_g = self._get_weight((self.cnn_feature_size, self.input_size))

        self.cb_i = self._get_weight((self.semantic_size, self.input_size))
        self.cb_f = self._get_weight((self.semantic_size, self.input_size))
        self.cb_o = self._get_weight((self.semantic_size, self.input_size))
        self.cb_g = self._get_weight((self.semantic_size, self.input_size))

        self.cc_i = self._get_weight((self.input_size, self.hidden_size))
        self.cc_f = self._get_weight((self.input_size, self.hidden_size))
        self.cc_o = self._get_weight((self.input_size, self.hidden_size))
        self.cc_g = self._get_weight((self.input_size, self.hidden_size))

        self.bias_i = self._get_bias((self.hidden_size, ))
        self.bias_f = self._get_bias((self.hidden_size, ))
        self.bias_o = self._get_bias((self.hidden_size, ))
        self.bias_g = self._get_bias((self.hidden_size, ))

        self.weight_dict = {
            'i': {
                'x': {
                    'a': self.wa_i,
                    'b': self.wb_i,
                    'c': self.wc_i
                },
                'v': {
                    'a': self.ca_i,
                    'b': self.cb_i,
                    'c': self.cc_i,
                },
                'h': {
                    'a': self.ua_i,
                    'b': self.ub_i,
                    'c': self.uc_i,
                }
            },
            'f': {
                'x': {
                    'a': self.wa_f,
                    'b': self.wb_f,
                    'c': self.wc_f,
                },
                'v': {
                    'a': self.ca_f,
                    'b': self.cb_f,
                    'c': self.cc_f,
                },
                'h': {
                    'a': self.ua_f,
                    'b': self.ub_f,
                    'c': self.uc_f,
                }
            },
            'o': {
                'x': {
                    'a': self.wa_o,
                    'b': self.wb_o,
                    'c': self.wc_o,
                },
                'v': {
                    'a': self.ca_o,
                    'b': self.cb_o,
                    'c': self.cc_o,
                },
                'h': {
                    'a': self.ua_o,
                    'b': self.ub_o,
                    'c': self.uc_o,
                }
            },
            'g': {
                'x': {
                    'a': self.wa_g,
                    'b': self.wb_g,
                    'c': self.wc_g,
                },
                'v': {
                    'a': self.ca_g,
                    'b': self.cb_g,
                    'c': self.cc_g,
                },
                'h': {
                    'a': self.ua_g,
                    'b': self.ub_g,
                    'c': self.uc_g,
                }
            },
        }

    def calculate_semantic_related_features(
        self,
        semantic: torch.Tensor,
        vector: torch.Tensor,
        gate_type: str,
        vector_type: str,
    ) -> torch.Tensor:
        """Calculate features with semantics incorporated based on the paper

        Args:
            semantic (torch.Tensor): (BATCH_SIZE, semantic_size)
            vector (torch.Tensor): (BATCH_SIZE, cnn_feature_size OR hidden_size OR embed_size)
            gate_type (torch.Tensor): [i, f, o, g]
            vector_type (torch.Tensor): [x, h, v]

        Returns:
            torch.Tensor: (BATCH_SIZE, hidden_size)
        """
        a_weight = self.weight_dict[gate_type][vector_type]['a']
        b_weight = self.weight_dict[gate_type][vector_type]['b']
        c_weight = self.weight_dict[gate_type][vector_type]['c']

        a_mul = torch.einsum('ij,bi->bj', a_weight, vector)
        b_mul = torch.einsum('ij,bi->bj', b_weight, semantic)
        hadamard = a_mul * b_mul
        c_mul = torch.einsum('ij,bi->bj', c_weight, hadamard)
        return c_mul

    def forward(self, captions: torch.Tensor, cnn_features: torch.Tensor, semantics: torch.Tensor) -> torch.Tensor:
        """Forward propagate

        Args:
            captions (torch.Tensor):  (BATCH_SIZE, caption_len)
            cnn_features (torch.Tensor):  (BATCH_SIZE, cnn_features_size)
            semantics (torch.Tensor):  (BATCH_SIZE, semantic_size)

        Returns:
            torch.Tensor:  (BATCH_SIZE, timestep-1, vocab_size)
        """
        batch_size, _ = captions.shape
        last_ht = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        last_ct = torch.zeros(batch_size, self.hidden_size).to(DEVICE)

        cnn_features = self.dropout_cnn(cnn_features)
        caption = captions[:, 0]

        result = torch.empty(batch_size, self.timestep - 1, self.vocab_size).to(DEVICE)
        for timestep_idx in range(self.timestep - 1):
            caption_embed = self.caption_embedding(caption)  # (BATCH_SIZE, embed_size)
            caption_embed = self.dropout_caption_embed(caption_embed)

            xi = self.calculate_semantic_related_features(semantics, caption_embed, 'i', 'x')
            xf = self.calculate_semantic_related_features(semantics, caption_embed, 'f', 'x')
            xo = self.calculate_semantic_related_features(semantics, caption_embed, 'o', 'x')
            xg = self.calculate_semantic_related_features(semantics, caption_embed, 'g', 'x')

            vi = self.calculate_semantic_related_features(semantics, cnn_features, 'i', 'v')
            vf = self.calculate_semantic_related_features(semantics, cnn_features, 'f', 'v')
            vo = self.calculate_semantic_related_features(semantics, cnn_features, 'o', 'v')
            vg = self.calculate_semantic_related_features(semantics, cnn_features, 'g', 'v')

            hi = self.calculate_semantic_related_features(semantics, last_ht, 'i', 'h')
            hf = self.calculate_semantic_related_features(semantics, last_ht, 'f', 'h')
            ho = self.calculate_semantic_related_features(semantics, last_ht, 'o', 'h')
            hg = self.calculate_semantic_related_features(semantics, last_ht, 'g', 'h')

            i_gate = self.sigmoid(xi + hi + vi + self.bias_i)
            f_gate = self.sigmoid(xf + hf + vf + self.bias_f)
            o_gate = self.sigmoid(xo + ho + vo + self.bias_o)
            g_gate = self.tanh(xg + hg + vg + self.bias_g)

            last_ct = f_gate * last_ct + i_gate * g_gate
            last_ht = o_gate * self.tanh(last_ct)

            out = self.linear_last(last_ht)  # (BATCH_SIZE, vocab_size)
            result[:, timestep_idx, :] = out
            caption = torch.argmax(out, dim=1)

        return result


if __name__ == '__main__':
    model = SemanticLSTM(
        input_size=1000,
        vocab_size=15000,
        hidden_size=500,
        embed_size=500,
        semantic_size=300,
        cnn_feature_size=4000,
    )
    model = model.cuda()

    batch_size = 8
    semantics = torch.randn((batch_size, 300)).cuda()
    cnn_features = torch.randn((batch_size, 4000)).cuda()
    captions = torch.ones((batch_size, 80)).long().cuda()

    res = model(captions, cnn_features, semantics)
    print(res.shape)
