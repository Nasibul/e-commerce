import pandas as pd
import datetime
from Classes import *

dummyitem = Item(234234, "ice cream", 'Chocolate', 6)
dummyitem2 = Item(3242443, "ice cream", 'Vanilla', 5)
dummyitem3 = Item(3222443, "ice cream", 'Strawberry', 6)
cool_store.restock(dummyitem, 10000)
cool_store.restock(dummyitem2, 10000)
print("Here are the items in the store")
cool_store.display_stock()
dummy = Customer("Nash", 24, "NYC")
dummy.grab(dummyitem, quantity=50)
dummy.grab(dummyitem2, quantity=50)
dummy.grab(dummyitem, quantity=100)
dummy.grab(dummyitem, quantity=100)
dummy.grab(dummyitem, quantity=100)
dummy.grab(dummyitem3, quantity=100)


cool_store.display_stock()

#dummy.return_item()