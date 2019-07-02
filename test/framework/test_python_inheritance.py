import unittest
import os

class TestPythonInheritance(unittest.TestCase):
    def test_class_dict(self):
        from flerken.framework.python_inheritance import ClassDict
        gt = {"office":
            {"medical": [
                {"room-number": 100,
                 "use": "reception",
                 "sq-ft": 50,
                 "price": 75
                 },
                {"room-number": 101,
                 "use": "waiting",
                 "sq-ft": 250,
                 "price": 75
                 },
                {"room-number": 102,
                 "use": "examination",
                 "sq-ft": 125,
                 "price": 150
                 },
                {"room-number": 103,
                 "use": "examination",
                 "sq-ft": 125,
                 "price": 150
                 },
                {"room-number": 104,
                 "use": "office",
                 "sq-ft": 150,
                 "price": 100
                 }
            ],
                "parking": {
                    "location": "premium",
                    "style": "covered",
                    "price": 750
                }
            }
        }
        subclassed = ClassDict(gt)
        path = './classdict.json'
        subclassed.write(path)
        loaded = ClassDict().load(path)

        self.assertTrue(os.path.isfile(path))
        self.assertEqual(subclassed, gt)
        self.assertEqual(loaded, gt)
        os.remove(path)
