import torch
from torch.nn import functional as F
from hydra.utils import instantiate


class CaptioningSolverTransformer:
    def __init__(self, model, n_epochs, optimizer,
                 train_loader, logger, device):
        self.device = device
        self.train_loader = instantiate(train_loader).loader

        self.model = instantiate(
            model, word_to_idx=self.train_loader.dataset.word_to_idx,
            input_dim=self.train_loader.dataset.features.shape[1]).to(self.device)

        self.data_iter = iter(self.train_loader)
        self.n_epochs = n_epochs

        self.print_every = logger.print_every
        self.verbose = logger.verbose
        self.optimizer = instantiate(optimizer, params=self.model.parameters())

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.loss_history = []

    def train_step(self):
        try:
            captions, features, urls = next(self.data_iter)
        except:
            self.data_iter = iter(self.train_loader)
            captions, features, urls = next(self.data_iter)

        captions_in = torch.tensor(captions[:, :-1],
                                   dtype=torch.long, device=self.device)
        captions_out = torch.tensor(captions[:, 1:],
                                    dtype=torch.long, device=self.device)
        mask = torch.tensor(captions_out != self.model._null,
                            dtype=torch.long, device=self.device)
        features = torch.tensor(features, device=self.device)

        logits = self.model(features, captions_in)
        loss = self.transformer_temporal_softmax_loss(logits, captions_out, mask)

        self.loss_history.append(loss.detach().cpu().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        n_iter = self.n_epochs * len(self.train_loader)
        for step in range(n_iter):
            self.train_step()
            if self.verbose and step % self.print_every == 0:
                print(f"(Iteration {step + 1} / {n_iter}) "
                      f"loss: {self.loss_history[-1]}")

    @staticmethod
    def transformer_temporal_softmax_loss(self, x, y, mask):
        batch_size, seq_len, vocab_size = x.size()

        x = x.view(batch_size * seq_len, -1)
        y = y.flatten()
        mask = mask.flatten()

        loss = F.cross_entropy(x,  y, reduction='none')
        loss = torch.mean(loss * mask)

        return loss
