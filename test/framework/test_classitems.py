import unittest
from flerken.framework import classitems
import torch
import numpy as np
from sklearn.metrics import accuracy_score


class TestClassItems(unittest.TestCase):
    def test_binary_accuracy_binary_predictions_BCE(self):
        gt = [torch.randint(0, 2, (3, 10)).float()]
        pred = [torch.randint(0, 2, (3, 10)).float()]

        gt_numpy = np.stack(gt).flatten()
        pred_numpy = (np.stack(pred).flatten() > 0.5).astype(float)

        accuracy_gt = accuracy_score(gt_numpy, pred_numpy)
        acc = classitems.TensorAccuracyItem(['Zero', 'One'])
        if isinstance(acc, classitems.TensorAccuracyItem):
            if acc.gt_transforms == 'auto':
                acc.predict_gt_transforms(torch.nn.BCELoss())
            if acc.pred_transforms == 'auto':
                acc.predict_pred_transforms(torch.nn.BCELoss())
        for x, y in zip(gt, pred):
            acc('train', x, y)

        acc.update_epoch('train')
        acc_results = acc.get_acc('train')
        self.assertEqual(accuracy_gt, acc_results)

    def test_binary_accuracy_probabilistic_predictions_BCE(self):
        gt = [torch.randint(0, 2, (3, 10)).float()]
        pred = [(torch.rand(3, 10).float() > 0.5).float()]

        gt_numpy = np.stack(gt).flatten()
        pred_numpy = np.stack(pred).flatten()

        accuracy_gt = accuracy_score(gt_numpy, pred_numpy)
        acc = classitems.TensorAccuracyItem(['Zero', 'One'])
        if isinstance(acc, classitems.TensorAccuracyItem):
            if acc.gt_transforms == 'auto':
                acc.predict_gt_transforms(torch.nn.BCEWithLogitsLoss())
            if acc.pred_transforms == 'auto':
                acc.predict_pred_transforms(torch.nn.BCEWithLogitsLoss())
        for x, y in zip(gt, pred):
            acc('train', x, y)

        acc.update_epoch('train')
        acc_results = acc.get_acc('train')
        self.assertEqual(accuracy_gt, acc_results)

    def test_binary_accuracy_binary_predictions_BCEwithlogits(self):
        gt = [torch.randint(0, 2, (3, 10)).float()]
        pred = [torch.randint(0, 2, (3, 10)).float()]

        gt_numpy = np.stack(gt).flatten()
        pred_numpy = (np.stack(pred).flatten() > 0.5).astype(float)

        pred = [torch.log(x / (1 - x)) for x in pred]
        accuracy_gt = accuracy_score(gt_numpy, pred_numpy)
        acc = classitems.TensorAccuracyItem(['Zero', 'One'])
        if isinstance(acc, classitems.TensorAccuracyItem):
            if acc.gt_transforms == 'auto':
                acc.predict_gt_transforms(torch.nn.BCEWithLogitsLoss())
            if acc.pred_transforms == 'auto':
                acc.predict_pred_transforms(torch.nn.BCEWithLogitsLoss())
        for x, y in zip(gt, pred):
            acc('train', x, y)

        acc.update_epoch('train')
        acc_results = acc.get_acc('train')
        self.assertEqual(accuracy_gt, acc_results)

    def test_multiclass_accuracy_CELoss(self):
        gt = [torch.randint(0, 2, (3, 1)).float()]
        pred = [torch.randint(-10, 10, (3, 3)).float()]
        pred_numpy = [torch.argmax(torch.softmax(x, dim=1), dim=1).float() for x in pred]

        gt_numpy = np.stack(gt).flatten()
        pred_numpy = (np.stack(pred_numpy).flatten()).astype(float)

        accuracy_gt = accuracy_score(gt_numpy, pred_numpy)
        acc = classitems.TensorAccuracyItem(['Zero', 'One', 'Two'])
        if isinstance(acc, classitems.TensorAccuracyItem):
            if acc.gt_transforms == 'auto':
                acc.predict_gt_transforms(torch.nn.CrossEntropyLoss())
            if acc.pred_transforms == 'auto':
                acc.predict_pred_transforms(torch.nn.CrossEntropyLoss())
        for x, y in zip(gt, pred):
            acc('train', x, y)

        acc.update_epoch('train')
        acc_results = acc.get_acc('train')
        self.assertEqual(accuracy_gt, acc_results)

    def test_multiclass_accuracy_NLLLoss(self):
        gt = [torch.randint(0, 2, (3, 1)).float()]
        pred = [torch.randint(-10, 10, (3, 3)).float()]
        pred = [torch.softmax(x, dim=1) for x in pred]
        pred_numpy = [torch.argmax(x, dim=1).float() for x in pred]

        gt_numpy = np.stack(gt).flatten()
        pred_numpy = (np.stack(pred_numpy).flatten()).astype(float)

        accuracy_gt = accuracy_score(gt_numpy, pred_numpy)
        acc = classitems.TensorAccuracyItem(['Zero', 'One', 'Two'])
        if isinstance(acc, classitems.TensorAccuracyItem):
            if acc.gt_transforms == 'auto':
                acc.predict_gt_transforms(torch.nn.NLLLoss())
            if acc.pred_transforms == 'auto':
                acc.predict_pred_transforms(torch.nn.NLLLoss())
        for x, y in zip(gt, pred):
            acc('train', x, y)

        acc.update_epoch('train')
        acc_results = acc.get_acc('train')
        self.assertEqual(accuracy_gt, acc_results)
