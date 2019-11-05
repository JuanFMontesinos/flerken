import unittest
from flerken.utils import ClassDict, Processor
import os


class TestClassDict(unittest.TestCase):
    def test_add_sucessful(self):
        a = ClassDict({1: 1})
        b = ClassDict({2: 1})
        gt = ClassDict({1: 1, 2: 1})
        assert gt == (a + b)

    def test_add_rep_key(self):
        with self.assertRaises(KeyError):
            a = ClassDict({1: 1})
            b = a + a

    def test_sum_sucess(self):
        a = ClassDict({1: 1})
        b = ClassDict({2: 1})
        c = ClassDict({3: 1})
        r = sum([a, b, c])

    def test_save_n_load(self):
        a = ClassDict({'1': 1})
        a.write('jamon.melon')
        b = ClassDict().load('jamon.json')
        try:
            assert a == b
            os.remove('jamon.json')
        except AssertionError as e:
            os.remove('jamon.json')
            raise e
