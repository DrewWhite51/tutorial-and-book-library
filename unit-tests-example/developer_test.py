import unittest
from unittest.mock import patch
from developer import Developer

# Write tests here


class TestDeveloper(unittest.TestCase):
    # These are for doing the setup and tear down one time for all tests
    # This is more often for things that are data intensive and only need to be set up once
    # @classmethod
    # def setUpClass(cls):
    #     print('setupClass')
    #
    # @classmethod
    # def tearDownClass(cls):
    #     print('teardownClass')

    # This happens at the beginning of every test.
    def setUp(self):
        self.dev1 = Developer('Drew', 'White', 'Python')
        self.dev2 = Developer('Rich', 'White', 'PHP')

    # This happens at the end of every test
    # Handy for creating objects such as DATABASES and files created during the test
    def tearDown(self):
        pass

    def test_email(self):

        self.assertEqual(self.dev1.email, 'Drew.White@gmail.com')
        self.assertEqual(self.dev2.email, 'Rich.White@gmail.com')

        self.dev1.last_name = 'Beast'
        self.dev2.last_name = 'Savage'

        self.assertEqual(self.dev1.email, 'Drew.Beast@gmail.com')
        self.assertEqual(self.dev2.email, 'Rich.Savage@gmail.com')

    def test_fullname(self):

        self.assertEqual(self.dev1.fullname, 'Drew White')
        self.assertEqual(self.dev2.fullname, 'Rich White')

    def test_give_raise(self):

        self.assertEqual(self.dev1.give_raise(), 110_000)
        self.assertEqual(self.dev2.give_raise(), 110_000)

        self.dev1.salary = 150_000
        self.dev2.salary = 170_000

        self.assertEqual(self.dev1.give_raise(), 160_000)
        self.assertEqual(self.dev2.give_raise(), 180_000)


