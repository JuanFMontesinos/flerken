import torch
from flerken.framework.pytorchframework import set_training, config
from flerken import pytorchfw
from flerken.framework import train, val, inference, test, ctx_iter,classitems


class toy_example(torch.nn.Module):
    def __init__(self):
        super(toy_example, self).__init__()
        self.module1 = torch.nn.Conv2d(1, 10, 3)
        self.module2 = torch.nn.Conv2d(10, 10, 3)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        return torch.nn.functional.sigmoid(x)

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

    def set_config(self):
        self.batch_size = 2
        self.dataparallel = False
        self.acc_ = classitems.TensorAccuracyItem(['perro', 'gato', 'casa', 'miau', 'guau'])

    @config
    @set_training
    def train(self):
        datab = database()
        self.train_loader = self.val_loader = torch.utils.data.DataLoader(datab, batch_size=self.batch_size)
        for self.epoch in range(self.start_epoch, self.EPOCHS):
            with train(self):
                self.run_epoch(self.train_iter_logger)
            with val(self):
                self.run_epoch()
