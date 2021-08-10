import torch
from torch.nn import functional as F
from hydra.utils import instantiate
from matplotlib import pyplot as plt

from image_utils import image_from_url


class CaptioningSolverTransformer:
    def __init__(self, model, n_epochs, optimizer,
                 train_loader, val_loader, logger, device):
        self.device = device
        self.train_loader = instantiate(train_loader).loader
        self.val_loader = instantiate(val_loader).loader

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

    def sample(self):
        max_iters = 1
        for i, (captions, features, urls) in enumerate(self.train_loader):
            if i >= max_iters:
                break
            captions = self.decode_captions(
                captions, idx_to_word=self.val_loader.dataset.idx_to_word)
            sample_captions = self.model.sample(features, max_len=30, device=self.device)
            sample_captions = self.decode_captions(
                sample_captions, self.train_loader.dataset.idx_to_word)
            for caption, sample_caption, url in zip(captions, sample_captions, urls):
                image = image_from_url(url)
                if image is not None:
                    plt.imshow(image)
                    plt.title(f"{sample_caption}\n{caption}")
                    plt.axis("off")
                    plt.show()


    @staticmethod
    def decode_captions(captions, idx_to_word):
        singleton = False
        if captions.ndim == 1:
            singleton = True
            captions = captions[None]
        decoded = []
        N, T = captions.shape
        for i in range(N):
            words = []
            for t in range(T):
                word = idx_to_word[captions[i, t]]
                if word != "<NULL>":
                    words.append(word)
                if word == "<END>":
                    break
            decoded.append(" ".join(words))
        if singleton:
            decoded = decoded[0]
        return decoded

    @staticmethod
    def transformer_temporal_softmax_loss(x, y, mask):
        batch_size, seq_len, vocab_size = x.size()

        x = x.view(batch_size * seq_len, -1)
        y = y.flatten()
        mask = mask.flatten()

        loss = F.cross_entropy(x,  y, reduction='none')
        loss = torch.mean(loss * mask)

        return loss
