import torch
from torch.nn.parameter import Parameter
from typing import Tuple


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
        label_size: int,
        cnn_feature_size: int,
        vocab_size: int,
        max_timestep: int = 80,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.label_size = label_size
        self.cnn_feature_size = cnn_feature_size
        self.max_timestep = max_timestep

        self._init_weights()
        self.word_embed = torch.nn.Embedding(vocab_size, embed_size)

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

        self.wb_i = self._get_weight((self.label_size, self.input_size))
        self.wb_f = self._get_weight((self.label_size, self.input_size))
        self.wb_o = self._get_weight((self.label_size, self.input_size))
        self.wb_g = self._get_weight((self.label_size, self.input_size))

        self.wc_i = self._get_weight((self.input_size, self.hidden_size))
        self.wc_f = self._get_weight((self.input_size, self.hidden_size))
        self.wc_o = self._get_weight((self.input_size, self.hidden_size))
        self.wc_g = self._get_weight((self.input_size, self.hidden_size))

        self.ua_i = self._get_weight((self.hidden_size, self.input_size))
        self.ua_f = self._get_weight((self.hidden_size, self.input_size))
        self.ua_o = self._get_weight((self.hidden_size, self.input_size))
        self.ua_g = self._get_weight((self.hidden_size, self.input_size))

        self.ub_i = self._get_weight((self.label_size, self.input_size))
        self.ub_f = self._get_weight((self.label_size, self.input_size))
        self.ub_o = self._get_weight((self.label_size, self.input_size))
        self.ub_g = self._get_weight((self.label_size, self.input_size))

        self.uc_i = self._get_weight((self.input_size, self.hidden_size))
        self.uc_f = self._get_weight((self.input_size, self.hidden_size))
        self.uc_o = self._get_weight((self.input_size, self.hidden_size))
        self.uc_g = self._get_weight((self.input_size, self.hidden_size))

        self.ca_i = self._get_weight((self.cnn_feature_size, self.input_size))
        self.ca_f = self._get_weight((self.cnn_feature_size, self.input_size))
        self.ca_o = self._get_weight((self.cnn_feature_size, self.input_size))
        self.ca_g = self._get_weight((self.cnn_feature_size, self.input_size))

        self.cb_i = self._get_weight((self.label_size, self.input_size))
        self.cb_f = self._get_weight((self.label_size, self.input_size))
        self.cb_o = self._get_weight((self.label_size, self.input_size))
        self.cb_g = self._get_weight((self.label_size, self.input_size))

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
            semantic (torch.Tensor): (BATCH_SIZE, label_size)
            vector (torch.Tensor): (BATCH_SIZE, cnn_feature_size OR hidden_size OR embed_size)
            gate_type (torch.Tensor): [i, f, o, g]
            vector_type (torch.Tensor): [x, h, v]

        Returns:
            torch.Tensor: (BATCH_SIZE, hidden_size)
        """
        a_weight = self.weight_dict[gate_type][vector_type]['a']
        b_weight = self.weight_dict[gate_type][vector_type]['b']
        c_weight = self.weight_dict[gate_type][vector_type]['c']

        a_mul = torch.einsum('bij,bi->bj', a_weight, vector)
        b_mul = torch.einsum('bij,bi->bj', b_weight, semantic)
        hadamard = a_mul * b_mul
        c_mul = torch.einsum('bij,bi->bj', c_weight, hadamard)
        return c_mul

    def forward(self, captions: torch.Tensor, cnn_features: torch.Tensor, semantics: torch.Tensor) -> torch.Tensor:
        """Forward propagate

        Args:
            captions (torch.Tensor): [description] (BATCH_SIZE, caption_len)
            cnn_features (torch.Tensor): [description] (BATCH_SIZE, cnn_features_size)
            semantics (torch.Tensor): [description] (BATCH_SIZE, label_size)

        Returns:
            torch.Tensor: [description]
        """
        batch_size, _ = captions.shape
        last_ht = torch.zeros(batch_size, self.hidden_size)
        last_ct = torch.zeros(batch_size, self.hidden_size)

        caption = captions[:, 0]

        for _ in range(self.max_timestep):
            word_embedded = self.word_embed(caption)  # (BATCH_SIZE, embed_size)

            xi = self.calculate_semantic_related_features(semantics, word_embedded, 'i', 'x')
            xf = self.calculate_semantic_related_features(semantics, word_embedded, 'f', 'x')
            xo = self.calculate_semantic_related_features(semantics, word_embedded, 'o', 'x')
            xg = self.calculate_semantic_related_features(semantics, word_embedded, 'g', 'x')

            vi = self.calculate_semantic_related_features(semantics, cnn_features, 'i', 'v')
            vf = self.calculate_semantic_related_features(semantics, cnn_features, 'f', 'v')
            vo = self.calculate_semantic_related_features(semantics, cnn_features, 'o', 'v')
            vg = self.calculate_semantic_related_features(semantics, cnn_features, 'g', 'v')

            hi = self.calculate_semantic_related_features(semantics, last_ht, 'i', 'h')
            hf = self.calculate_semantic_related_features(semantics, last_ht, 'f', 'h')
            ho = self.calculate_semantic_related_features(semantics, last_ht, 'o', 'h')
            hg = self.calculate_semantic_related_features(semantics, last_ht, 'g', 'h')

            i_gate = torch.nn.Sigmoid(xi + hi + vi + self.bias_i)
            f_gate = torch.nn.Sigmoid(xf + hf + vf + self.bias_f)
            o_gate = torch.nn.Sigmoid(xo + ho + vo + self.bias_o)
            g_gate = torch.nn.Tanh(xg + hg + vg + self.bias_g)

            last_ct = f_gate * last_ct + i_gate * g_gate
            last_ht = o_gate * torch.nn.Tanh(last_ct)


if __name__ == '__main__':
    model = SemanticLSTM(input_size=1000, hidden_size=500, embed_size=500, label_size=300, cnn_feature_size=4000)
    model = model.cuda()
    print(model)
