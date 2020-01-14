import torch
from numpy import nan, inf
from flerken.framework.pytorchframework import set_training, config
from flerken import pytorchfw
from flerken.framework import train, val, inference, test, ctx_iter, classitems


class toy_example(torch.nn.Module):
    def __init__(self, isnan=False, isinf=False):
        super(toy_example, self).__init__()
        self.module1 = torch.nn.Conv2d(1, 10, 3)
        self.module2 = torch.nn.Conv2d(10, 10, 3)
        self.isnan = isnan
        self.isinf = isinf

    def forward(self, x):
        x = self.module1(x)
        if self.isnan:
            x += torch.tensor(nan)

        x = self.module2(x)
        if self.isinf:
            x += torch.tensor(inf)
        return torch.sigmoid(x)


class database(torch.utils.data.Dataset):
    def __len__(self):
        return 30

    def __getitem__(self, idx):
        return torch.randint(0, 2, (10, 6, 6)).float(), [torch.randint(0, 5, (1, 10, 10)).float()], []


class toy_fw(pytorchfw):
    def hyperparameters(self):
        self.hihihi = 5
        self.initializer = 'xavier'
        self.EPOCHS = 1
        self.LR = 0.000000000001
        # Def optimizer self.optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR)
        # Def criterion self.criterion
        self.criterion = torch.nn.BCELoss().to(self.main_device)
        self.init_acc(['perro', 'gato', 'casa', 'miau', 'guau'], False)

    def set_config(self):
        self.batch_size = 2
        self.dataparallel = False

    @config
    @set_training
    def train(self):
        datab = database()
        self.train_loader = self.val_loader = torch.utils.data.DataLoader(datab, batch_size=self.batch_size)
        for self.epoch in range(self.start_epoch, self.EPOCHS):
            with train(self):
                self.run_epoch(self.train_loader, metrics=['acc'])
            with val(self):
                with torch.no_grad():
                    self.run_epoch(self.train_loader, metrics=['acc'], checkpoint=self.checkpoint(metric='acc',
                                                                                                  criteria=[max, lambda
                                                                                                      epoch,
                                                                                                      optimal: optimal > epoch]))
