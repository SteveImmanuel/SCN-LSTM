import torch
import torchvision.models as models
from typing import Tuple, Dict
from s2vt.constant import *


class S2VT(torch.nn.Module):
    def __init__(
            self,
            word_to_idx: Dict,
            vocab_size: int,
            timestep: int = 80,
            lstm_input_size: int = 500,
            lstm_hidden_size: int = 500,
            drop_out_rate: float = .3,
            cnn_out_size: int = 4096,  # depends on the pretrained CNN architecture
    ) -> None:
        super().__init__()
        self.word_to_idx = word_to_idx
        self.timestep = timestep
        self.vocab_size = vocab_size
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn_extractor = self._initialize_vgg()

        self.first_lstm = torch.nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )

        self.second_lstm = torch.nn.LSTM(
            input_size=lstm_input_size + lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )

        self.caption_embedding = torch.nn.Embedding(vocab_size, lstm_hidden_size)
        self.video_embedding = torch.nn.Linear(cnn_out_size, lstm_input_size)
        self.linear_last = torch.nn.Linear(lstm_hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(drop_out_rate)
        self.relu = torch.nn.ReLU()

    def _initialize_vgg(self):
        vgg = models.vgg16(pretrained=True).to(DEVICE)
        for param in vgg.parameters():
            param.requires_grad = False
        # use output from fc7, remove the rest
        vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])
        return vgg

    def forward(self, data: Tuple[torch.Tensor, int], caption: torch.Tensor = None) -> torch.Tensor:
        """Forward propagate

        Args:
            x (torch.Tensor): (BATCH_SIZE, timestep, channel, height, width)
            caption (torch.Tensor): (BATCH_SIZE, timestep)
        Returns:
            model inference result 
                if training, raw unnormalized scores for each class (BATCH_SIZE, timestep, vocab_size)
                else (BATCH_SIZE, timestep, vocab_size)
        """
        x, image_seq_len = data
        batch_size, _, _, _, _ = x.shape
        extracted_features = []
        for i in range(self.timestep):
            temp_extracted = self.cnn_extractor(x[:, i, :, :, :])
            temp_extracted = self.video_embedding(temp_extracted)
            extracted_features.append(temp_extracted)

        extracted_features = torch.cat(extracted_features, 0).to(DEVICE)
        extracted_features = extracted_features.view(batch_size, self.timestep, -1)
        # output from CNN extractor (BATCH_SIZE, timestep, lstm_input_size)
        x, _ = self.first_lstm(extracted_features)

        if self.training:
            # right shift by one for concatenating output from first lstm
            caption = caption[:, :-1]
            pad_zero = torch.zeros(caption.shape[0], 1).to(DEVICE).long()
            caption = torch.cat((pad_zero, caption), 1).to(DEVICE)
            # dim (BATCH_SIZE, timestep, lstm_hidden_size)
            caption = self.caption_embedding(caption)

            # concatenate output from first lstm with caption
            x = torch.cat((x, caption), 2).to(DEVICE)
            x, _ = self.second_lstm(x)
            x = self.dropout(x)
            x = self.linear_last(x)
            x = self.relu(x)
            return x
        else:
            pad_zero = torch.zeros(batch_size, image_seq_len, self.lstm_hidden_size).to(DEVICE).long()
            encode_input = torch.cat((x[:, :image_seq_len, :], pad_zero), 2).to(DEVICE)

            final_output = torch.zeros(batch_size, self.timestep).to(DEVICE).long()
            final_output[:, 0] = self.word_to_idx[BOS_TAG]
            final_output[:, -1] = self.word_to_idx[EOS_TAG]

            _, (hn, cn) = self.second_lstm(encode_input)
            caption_out = self.word_to_idx[BOS_TAG] * torch.ones(batch_size, 1).to(DEVICE).long()

            for i in range(self.timestep - 1 - image_seq_len):  # timestep minus <BOS>, <EOS>, and image_seq_len
                caption_out = self.caption_embedding(caption_out)
                decode_input = torch.cat((x[:, image_seq_len + i + 1, :], caption_out), 2).to(DEVICE)

                result, (hn, cn) = self.second_lstm(decode_input, (hn, cn))
                result = self.dropout(result)
                result = self.linear_last(result)
                result = self.relu(result)  # (BATCH_SIZE, 1, vocab_size)

                caption_out = torch.argmax(result, dim=2).to(DEVICE).long()
                final_output[:, i + 1] = caption_out
            return final_output
