import torch
from torch import nn


class JuryNet(nn.Module):
    def __init__(
        self, input_dim=768, num_juries=128, hidden_dropout=0.1, output_dropout=0.7
    ):
        super(JuryNet, self).__init__()
        self.num_juries = num_juries
        self.input_dim = input_dim
        self.hidden_dropout = hidden_dropout
        self.output_dropout = output_dropout
        self.loss_fn = nn.BCELoss(reduction="none")
        self.classifier = nn.Sequential(
            self.init_linear(self.input_dim, 512),
            nn.Dropout(p=self.hidden_dropout),
            nn.ReLU(),
            self.init_linear(512, 256),
            nn.Dropout(p=self.hidden_dropout),
            nn.ReLU(),
            self.init_linear(256, num_juries),  # load and initialize weights
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(params=self.classifier.parameters(), lr=0.001)

    def init_linear(self, in_size, out_size):
        layer = nn.Linear(in_size, out_size)
        torch.nn.init.kaiming_uniform_(layer.weight)
        return layer

    def get_discourse(self, pred, discourse_type="std"):
        """

        Args:
          discourse_type: one of {'std'}


        Returns: tensor with discourse of final output layer of forward pass for each entry in the batch.

        """
        if discourse_type == "entropy":
            raise NotImplementedError("Not yet implemented. Please used 'std' for now.")
        elif discourse_type == "std":
            return torch.std(pred, dim=-1)
        else:
            raise Exception("discourse_type must be one of {'std'}")

    def uncertainty(self, input, discourse_type="std"):
        pred = self.forward(input)
        return self.get_discourse(pred, discourse_type=discourse_type)

    def loss(self, preds, labels):
        """
        labels: [batch_size, 1]
        preds: [batch_size, num_juries]
        """
        mask_tensor = torch.full(
            size=(1, self.num_juries), fill_value=self.output_dropout, device=device
        )
        mask = torch.bernoulli(mask_tensor)
        # masked_preds = preds * mask
        labels = labels.unsqueeze(1).expand(preds.shape)
        # masked_labels = preds * labels
        loss_result = self.loss_fn(preds.float(), labels.float())
        loss_result = (loss_result * mask).sum() / mask.sum()

        return loss_result

    def forward(self, input):
        """
        Args:
          inputs: Tensor from BERT of shape [batch_size, 768]

        Output:

        """
        return self.classifier(input)
