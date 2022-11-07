import pandas as pd
import datetime
from Classes import *
from config import *

# TODO 

# multi decorator validation for grab
# make gen stock secure
# log should be list 
# unitest each method - reference test time buy method = 1 min 25 secs
# call buy method seperately from command line
# git hook that runs tests on code push
# main function
# write agent
# create task agent
# test driven development

cool_store = Store(location="NYC")
cool_store.gen_stock(path='items.csv')

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

while sku != 999:
    if sku in cool_store.stock[SKU_ID]:
        quantity = int(input('Quantity?\n'))
        row = cool_store.stock[cool_store.stock[SKU_ID]==sku].iloc[0]
        item = Item(sku=sku, name= row["Name"], description=row['Description'], price=int(row['Price']))
        if quantity > row["Quantity"]:
            print('''We do not have not much of this particular item.
Please enter another item or a lesser quantity of this item.''')
        else:
            dummy.grab(item=item, quantity=quantity, store=cool_store)
            print('Added to cart')
        sku = int(input())
    elif sku != 0: 
        print('We do not have that item. Please enter a SKU')
        sku = int(input())
    elif sku == 0:
        return_sku = int(input('Item SKU?\n'))
        if return_sku in cool_store.stock[SKU_ID]:
            return_quantity = int(input('Item quantity?\n'))
            return_item_row = cool_store.stock[cool_store.stock[SKU_ID]==return_sku].index[0]
            cool_store.stock.at[return_item_row, "Quantity"] += return_quantity
            print(f"Item returned to store")
        sku = int(input())

print('\n')
check_out = dummy.buy(store=cool_store)
print(check_out.name)
print(check_out)
print(*check_out.list, sep='\n')
print(f'Total is ${"{:.2f}".format(check_out.total)}')

