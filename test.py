import unittest 
from Classes import *
from config import *

class Tester(unittest.TestCase):

    def test_grab(self):
        dummy = Customer(name='Nasibul', age=24, location='NYC')
        dummy_store = Store(location="NYC")
        dummy_store.gen_stock(filepath='items.csv')
        dummy.grab(sku='33r', store=dummy_store, quantity=1000000)
        self.assertEqual(len(dummy.shopping_cart), 0)
        dummy.grab(sku='33r', store=dummy_store, quantity=10)
        self.assertEqual(len(dummy.shopping_cart), 1)
        
if __name__ == '__main__':
    unittest.main()