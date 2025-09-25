# This file extends the functionality of the original 
# O5Synthetic and O5Dataset classes (n=5) to the case of arbitrary n
# Original code from: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks


import numpy as np
from torch.utils.data import DataLoader



class OnSynthetic(object):
    def __init__(self, d, N=1024):
        super().__init__()
        self.dim = 2 * d

        # generate vector parts
        X_core = np.random.randn(N, self.dim)
        ri = X_core.reshape(-1, 2, d)
        r1, r2 = ri.transpose(1, 0, 2)

        # generate scalar parts
        scalars = np.random.randn(N, 2)
        sc1, sc2 = scalars[:, 0], scalars[:, 1]

        # generate pseudoscalar parts
        pseudoscalars = np.random.randn(N, 2)
        ps1, ps2 = pseudoscalars[:, 0], pseudoscalars[:, 1]

        # Full X now includes vectors r1,r2 and the extra scalars sc1,sc2 and pseudoscalars ps1, ps2
        self.X = np.concatenate([X_core, scalars, pseudoscalars], axis=1)  # shape (N, 2*d+4) (sc1, sc2, ps1, ps2 are the last 4 entities)

        self.Y = (
            sc1 * np.sin(np.sqrt((r1**2).sum(-1)))
            - 0.5 * np.sqrt((r2**2).sum(-1)) ** 3
            + (sc1 * sc2) * (r1 * r2).sum(-1)
            / (np.sqrt((r1**2).sum(-1)) * np.sqrt((r2**2).sum(-1)))
            + np.arctan(np.sqrt((ps1**2).sum(-1)))
        )
        self.Y = self.Y[..., None]

        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)  # can add and subtract arbitrary tensors
        Xscale = (
            np.sqrt((X_core.reshape(N, 2, d) ** 2).mean((0, 2)))[:, None] + 0 * ri[0]
        ).reshape(self.dim)
        Xscale = np.concatenate([Xscale, np.ones(4)], axis=0)  # add scale for c1, c2, ps1, ps2
        self.stats = Xmean, Xscale, self.Y.mean(axis=0), self.Y.std(axis=0)

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]


class OnDataset:
    def __init__(self, n, num_train_samples=1000, num_test_samples=100, batch_size=16): # in our O(8)-Regression, we have n=8
        super().__init__()
        self.train_dataset = OnSynthetic(d=n, N=num_train_samples)
        self.val_dataset = OnSynthetic(d=n, N=num_test_samples)
        self.test_dataset = OnSynthetic(d=n, N=num_test_samples)

        self.batch_size = batch_size

        self.ymean, self.ystd = self.train_dataset.stats[-2].item(), self.train_dataset.stats[-1].item()

        self._normalize_datasets()

    def _normalize_datasets(self):
        Xmean, Xscale, Ymean, Ystd = self.train_dataset.stats
        self.train_dataset.X -= Xmean
        self.train_dataset.X /= Xscale
        self.train_dataset.Y -= Ymean
        self.train_dataset.Y /= Ystd

        self.val_dataset.X -= Xmean
        self.val_dataset.X /= Xscale
        self.val_dataset.Y -= Ymean
        self.val_dataset.Y /= Ystd

        self.test_dataset.X -= Xmean
        self.test_dataset.X /= Xscale
        self.test_dataset.Y -= Ymean
        self.test_dataset.Y /= Ystd

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )