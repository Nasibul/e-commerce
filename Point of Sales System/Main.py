import pandas as pd
import datetime
from Classes import *
from config import *
import random
from Settings import *

# TODO

# log should be list
# unitest each method - reference test time buy method = 1 min 25 secs
# jupyter notebook of log data
# call buy method seperately from command line
# git hook that runs tests on code push
# main function
# write agent
# create task agent
# test driven development

cool_store = Store(location="NYC")
cool_store.gen_stock(filepath='items.csv')

# if statement for choosing manual and auto
control = input("Manual or Auto?\n")
assert control == 'Manual' or control == 'Auto', f'Please make a valid choice'


def manual():
    print(f"Welcome to the {cool_store.location} store!")
    name = input("Please enter your name\n")
    age = int(input('Age?\n'))
    location = input('Location?\n')
    dummy = Customer(name=name, age=age, location=location)
    print(
        '''Here is our stock. To add an item to your cart, please type the item SKU and press enter.
    To finish grabbing items, type 999 to check out.
    To return an item, type 0''', '\n')
    cool_store.display_stock()
    sku = int(input())
    while sku != 999:
        if sku == 0:
            quantity = 0
        else:
            quantity = int(input('Quantity?\n'))
            while quantity <= 0:
                quantity = int(input('Please enter an actual amount\n'))
        dummy.grab(sku=sku, quantity=quantity, store=cool_store)
        sku = int(input())

    print('\n')
    check_out = dummy.buy(store=cool_store)
    print(check_out.name)
    print(check_out)
    print(*check_out.list, sep='\n')
    print(f'Total is ${"{:.2f}".format(check_out.total)}')


def run_agent(bot_number):
    name = 'Bot #' + str(bot_number+1)
    age = random.randint(18, 80)
    location = random.choice(LOCATIONS)
    bot = Customer(name=name, age=age, location=location)
    shopping_trip = random.randint(1, 10)
    for i in range(0, shopping_trip):
        sku = random.randint(1, 33)
        quantity = random.randint(1, 10)
        bot.grab(sku=sku, quantity=quantity, store=cool_store)

    print('\n')
    check_out = bot.buy(store=cool_store)
    print(check_out.name)
    print(check_out)
    print(*check_out.list, sep='\n')
    print(f'Total is ${"{:.2f}".format(check_out.total)}')


def run_simulation(agents):
    for i in range(0, agents):
        run_agent(i)
    df = pd.DataFrame(cool_store.log)
    df.to_csv('output.csv')


if control == 'Manual':
    manual()
else:
    run_simulation(100000)
