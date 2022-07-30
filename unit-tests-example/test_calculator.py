import unittest
import calculator


class TestCalculator(unittest.TestCase):

    def test_add(self):
        self.assertEqual(calculator.add(10, 5), 15)
        self.assertEqual(calculator.add(-6, 3), -3)
        self.assertEqual(calculator.add(-9, -8), -17)

    def test_multiply(self):
        self.assertEqual(calculator.multiply(5, 10), 50)
        self.assertEqual(calculator.multiply(2.5, 3.5), 8.75)

    def test_subtract(self):
        self.assertEqual(calculator.subtract(20, 13), 7)

    def test_divide(self):
        self.assertEqual(calculator.divide(32, 8), 4)
        # This asserts that the value cannot be divided by 0
        self.assertRaises(ValueError, calculator.divide, 10, 0)
        # Exact same as above but uses a context manager instead
        with self.assertRaises(ValueError):
            calculator.divide(10, 0)


if __name__ == '__main__':
    unittest.main()
