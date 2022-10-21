import pandas as pd
import datetime
import copy
from Classes import *

# TODO 
# .index remove through stock as dataframe
# get rid of prints
# multi wrapper validate stock + validate transaction
# unitest each method - reference test time buy method = 1 min 25 secs
# git hook that runs tests on code push
# spawn store function
# spawn customer - real or agent
# main function
# write agent
# create task agent

cool_store = Store(location="NYC")
items_source = pd.read_csv('items.csv')
items_source.index +=1
for i in range(len(items_source)):
    sku = int(i)
    name = items_source.iloc[i]['Name']
    description = items_source.iloc[i]['Description']
    price = float(items_source.iloc[i]['Price'])
    item = Item(sku=sku, name=name, description=description, price=price)
    cool_store.restock(item=item, quantity=50)

print(f"Welcome to the {cool_store.location} store!")
name = input("Please enter your name\n")
age = int(input('Age?\n'))
location = input('Location?\n')
dummy = Customer(name=name, age=age, location=location)
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
        dummy.grab(item=item, quantity=quantity, store=cool_store)
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
        dummy.return_item(item=return_item, quantity=return_quantity, store=cool_store)
        sku = int(input())

print('\n')
check_out = dummy.buy(store=cool_store)
print(check_out.name)
print(check_out)
print(*check_out.list, sep='\n')
print(f'Total is ${"{:.2f}".format(check_out.total)}')

