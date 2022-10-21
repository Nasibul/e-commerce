import pandas as pd
import datetime
import copy
from Classes import *

# TODO 
# fix code
# tx log store
# column names global vars
# validate transaction wrappers
# spawn store function
# spawn customer - real or agent
# main function
# write agent
# create task agent

cool_store = Store("NYC")
tx_log = pd.DataFrame(columns=["Customer Name", "Transaction Date and Time", "Customer Age",
                            "Cart", "Number of Items", "Discount", "Total"])
items_source = pd.read_csv('items.csv')
items_source.index +=1
for i in range(len(items_source)):
    sku = int(i)
    name = items_source.iloc[i]['Name']
    description = items_source.iloc[i]['Description']
    price = float(items_source.iloc[i]['Price'])
    item = Item(sku, name, description, price)
    cool_store.restock(item, 50)

print(f"Welcome to the {cool_store.location} store!")
name = input("Please enter your name\n")
age = int(input('Age?\n'))
location = input('Location?\n')
dummy = Customer(name, age, location)
print(
'''Here is our stock. To add an item to your cart, please type the item SKU and press enter.
To finish grabbing items, type 999 to check out.
To return an item, type 0''','\n')
cool_store.display_stock()
sku = int(input())
all_sku = [i.sku for i in cool_store.stock]

while sku != 999:
    if sku in all_sku:
        quantity = int(input('Quantity?\n'))
        index = all_sku.index(sku)
        item = cool_store.stock[index]
        dummy.grab(item, quantity, cool_store)
        print('Added to cart')
        sku = int(input())
    elif sku != 0: 
        print('We do not have that item. Please enter a SKU')
        sku = int(input())
    elif sku == 0:
        return_sku = int(input('Item SKU?\n'))
        return_quantity = int(input('Item quantity?\n'))
        return_index = all_sku.index(return_sku)
        return_item = cool_store.stock[return_index]
        dummy.return_item(return_item, return_quantity, cool_store)
        sku = int(input())

print('\n')
dummy.buy()

