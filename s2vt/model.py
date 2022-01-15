import torch
import torchvision.models as models


class S2VT(torch.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            timestep: int = 80,
            lstm_input_size: int = 500,
            lstm_hidden_size: int = 500,
            drop_out_rate: float = .3,
            cnn_out_size: int = 4096,  # depends on the pretrained CNN architecture
    ) -> None:
        super().__init__()
        self.timestep = timestep
        self.vocab_size = vocab_size
        self.lstm_input_size = lstm_input_size
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
        self.softmax = torch.nn.Softmax(dim=2)

    def _initialize_vgg(self):
        vgg = models.vgg16(pretrained=True).cuda()
        for param in vgg.parameters():
            param.requires_grad = False
        # use output from fc6, remove the rest
        vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:4])
        return vgg

    def forward(self, x: torch.Tensor, caption: torch.Tensor = None) -> torch.Tensor:
        """Forward propagate

        Args:
            x (torch.Tensor): (BATCH_SIZE, image_sequence_length, channel, height, width)
            caption (torch.Tensor): (BATCH_SIZE, timestep, vocab_size)
        Returns:
            model inference result 
                if training (BATCH_SIZE, timestep, vocab_size)
                else (BATCH_SIZE, length until EOS tag emitted, vocab_size)
        """
        batch_size, image_seq_len, _, _, _ = x.shape
        extracted_features = []
        for i in range(x.shape[1]):
            temp_extracted = self.cnn_extractor(x[:, i, :, :, :])
            temp_extracted = self.video_embedding(temp_extracted)
            extracted_features.append(temp_extracted)

        extracted_features = torch.cat(extracted_features, 0)
        extracted_features = extracted_features.view(batch_size, image_seq_len, -1)
        # output from CNN extractor (BATCH_SIZE, image_sequence_length, lstm_input_size)

        if self.training:
            # pad with zero for the remaining timestep
            pad_zero = torch.zeros(batch_size, self.timestep - image_seq_len, self.lstm_input_size).cuda()
            x = torch.cat((extracted_features, pad_zero), 1).cuda()

            x, _ = self.first_lstm(x)

            # right shift by one for concatenating output from first lstm
            caption = caption[:, :-1]
            pad_zero = torch.zeros(caption.shape[0], 1).cuda().long()
            caption = torch.cat((pad_zero, caption), 1).cuda()
            # dim (BATCH_SIZE, timestep, lstm_hidden_size)
            caption = self.caption_embedding(caption)

            # concatenate output from first lstm with caption
            x = torch.cat((x, caption), 2)
            x, _ = self.second_lstm(x)
            x = self.dropout(x)
            x = self.linear_last(x)
            return self.softmax(x)
        else:
            raise NotImplementedError()
        # image_dim = image_seq[0].shape
        # # pad with zero for the remaining timestep
        # pad_zero = torch.zeros((self.timestep - image_seq_len, *image_dim[1:]))
        # image_seq.append(pad_zero)

        # pad ending annotation with <BOS> until the length matches timestep
        # annot_padded = annot_raw + [BOS_TAG] * (self.timestep - len(annot_raw) - (image_seq_len - 1))

        # # pad beggining with zero
        # pad_zero = torch.zeros((image_seq_len - 1, self.vocab_size))
        # # output dim = (timestep, vocab_size)
        # label_annotation = torch.cat([pad_zero, label_annotation], 0)

        # image_dim = image_seq[0].shape
        # # pad with zero for the remaining timestep
        # pad_zero = torch.zeros((self.timestep - image_seq_len, *image_dim[1:]))
        # image_seq.append(pad_zero)