import unittest

if __name__ == '__main__':
    testsuite = unittest.TestLoader().discover('./test')
    unittest.TextTestRunner(verbosity=1).run(testsuite)