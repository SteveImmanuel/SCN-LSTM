import torch
import torchvision.models as models
from typing import Tuple, Dict
from s2vt.constant import *


class S2VT(torch.nn.Module):
    def __init__(
            self,
            word_to_idx: Dict,
            vocab_size: int,
            timestep: int = 80,  # each video and annotation will have timestep sequence, so model will process 2*timestep
            video_embed_size: int = 500,
            caption_embed_size: int = 500,
            lstm_hidden_size: int = 500,
            drop_out_rate: float = .3,
            cnn_out_size: int = 4096,  # depends on the pretrained CNN architecture
    ) -> None:
        super().__init__()
        self.word_to_idx = word_to_idx
        self.timestep = timestep
        self.vocab_size = vocab_size
        self.video_embed_size = video_embed_size
        self.caption_embed_size = caption_embed_size

        self.first_lstm = torch.nn.LSTM(
            input_size=video_embed_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )

        self.second_lstm = torch.nn.LSTM(
            input_size=caption_embed_size + lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )

        self.caption_embedding = torch.nn.Embedding(vocab_size, caption_embed_size)
        self.video_embedding = torch.nn.Linear(cnn_out_size, video_embed_size)
        self.linear_last = torch.nn.Linear(lstm_hidden_size, vocab_size)
        self.dropout_video_embed = torch.nn.Dropout(drop_out_rate)
        self.dropout_caption_embed = torch.nn.Dropout(drop_out_rate)

    def forward(self, data: torch.Tensor, caption: torch.Tensor = None) -> torch.Tensor:
        """Forward propagate

        Args:
            data (torch.Tensor): (BATCH_SIZE, timestep, cnn_out_size)
            caption (torch.Tensor): (BATCH_SIZE, timestep)
        Returns:
            model inference result 
            raw unnormalized scores for each class (BATCH_SIZE, timestep, vocab_size)
        """
        batch_size, _, _ = data.shape
        extracted_features = self.video_embedding(data)
        extracted_features = self.dropout_video_embed(extracted_features)  # (BATCH_SIZE, timestep, video_embed_size)
        pad_zero = torch.zeros(batch_size, self.timestep, self.video_embed_size).to(DEVICE)

        lstm1_in = torch.cat((extracted_features, pad_zero), dim=1).to(DEVICE)
        lstm1_out, _ = self.first_lstm(lstm1_in)  # (BATCH_SIZE, 2*timestep, lstm_hidden_size)

        final_out = torch.zeros(batch_size, self.timestep, self.vocab_size).to(DEVICE)

        lstm2_state = None

        for current_timestep in range(2 * self.timestep):
            lstm2_in = lstm1_out[:, current_timestep:current_timestep + 1, :]  # (BATCH_SIZE, 1, lstm_hidden_size)

            if current_timestep < self.timestep:
                caption_out = torch.zeros(batch_size, 1, self.caption_embed_size).long().to(DEVICE)
            else:
                if current_timestep == self.timestep:
                    caption = torch.ones(batch_size, 1).long().to(DEVICE) * self.word_to_idx[BOS_TAG]

                caption_out = self.caption_embedding(caption)
                caption_out = self.dropout_caption_embed(caption_out)

            # (BATCH_SIZE, 1, lstm_hidden_size+caption_embed_size)
            lstm2_in = torch.cat((lstm2_in, caption_out), axis=2)

            if current_timestep == 0:
                lstm2_out, lstm2_state = self.second_lstm(lstm2_in)
            else:
                lstm2_out, lstm2_state = self.second_lstm(lstm2_in, lstm2_state)

            raw_caption = self.linear_last(lstm2_out)  # (BATCH_SIZE, 1, vocab_size)
            caption = torch.argmax(raw_caption, axis=2).to(DEVICE).long()

            if (current_timestep >= self.timestep):
                idx = current_timestep - self.timestep
                final_out[:, idx:idx + 1, :] = raw_caption

        return final_out
