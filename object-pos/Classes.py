import datetime
from unittest import result
import pandas as pd
import config as cnf
import random
from randomtimestamp import randomtimestamp
import os.path


def clear_cart(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        args[0].shopping_cart = []
        return result
    return wrapper
# validate return option


def return_option(func):
    def wrapper(*args, **kwargs):
        if kwargs['sku'] == 0:
            return_sku = int(input('Return Item SKU?\n'))
            while return_sku <= 0:
                return_sku = int(input('Return Item SKU?\n'))
            return_quantity = int(input('Item quantity?\n'))
            while return_quantity <= 0:
                return_quantity = int(input('Please enter an actual amount\n'))
            return_item_row = kwargs['store'].stock[kwargs['store']
                                                    .stock[cnf.SKU_ID] == return_sku].index[0]
            kwargs['store'].stock.at[return_item_row,
                                     "Quantity"] += return_quantity
            print(f"Item returned to store")
        func(*args, **kwargs)
    return wrapper


def existance_quantity(func):
    def wrapper(*args, **kwargs):
        try:
            row = kwargs['store'].stock[kwargs['store'].stock[cnf.SKU_ID]
                                        == kwargs['sku']].iloc[0]
            if kwargs['quantity'] > int(row["Quantity"]):
                print('''We do not have not much of this particular item.
Please enter another item or the same sku and a lesser quantity of this item.''')
            else:
                func(*args, **kwargs)
        except:
            if kwargs['sku'] == 0:
                pass
            else:
                print('We do not have that item. Please enter a SKU')
    return wrapper


class Item:
    def __init__(self, sku: int, name: str, description: str, price: float):
        if type(price) == int:
            price = float(price)
        assert type(
            name) == str, f"Name must be a string, instead found {name}"
        assert type(
            description) == str, f"description must be a string, instead found {description}"
        assert type(
            price) == float, f"Price must be a float, instead found {price}"
        self.sku = sku
        self.name = name.title()
        self.description = description
        self.price = round(price, 2)

    def __str__(self):
        return f'SKU#{self.sku}, {self.name}, {self.description}, ${"{:.2f}".format(self.price)}, Quantity= {self.quantity}'

    def __repr__(self):
        return str(self)


class Store:
    def __init__(self, location: str):
        assert type(
            location) == str, f"Location must be a string, instead found {location}"
        self.location = location
        self.log = []

    def gen_stock(self, filepath: str):
        assert os.path.exists(filepath) == True, "File not found"
        temp = pd.read_csv(filepath)
        assert cnf.ITEM_PRICE == temp.columns[2], "Column not found"
        assert cnf.ITEM_NAME == temp.columns[0], "Column not found"
        assert cnf.ITEM_DESCRIPTION == temp.columns[1], "Column not found"
        self.stock = pd.DataFrame({
            cnf.ITEM_PRICE: temp["Price"],
            cnf.ITEM_NAME: temp['Name'],
            cnf.ITEM_DESCRIPTION: temp['Description'],
        })
        self.stock[cnf.SKU_ID] = [1 + int(i) for i in temp.index]
        self.stock[cnf.QUANTITY] = [random.randint(13000, 19000) for i in temp.index]

    def display_stock(self):
        print(self.stock)

    def __str__(self):
        return f'{self.location} Store'


class Customer:
    def __init__(self, name: str, age: int, location: str):
        assert type(name) == str, f"Name must be a string, instead found {name}"
        assert type(age) == int, f"Age must be an integer, instead found {age}"
        assert type(location) == str, f"Location must be a string, instead found {location}"
        self.name = name
        self.age = age
        self.location = location
        self.shopping_cart = []

    def __str__(self):
        return f'{self.name}, {self.age}, {self.location}'

    @return_option
    @existance_quantity
    def grab(self, sku: int, store: Store, quantity: int):
        row = store.stock[store.stock[cnf.SKU_ID] == sku].iloc[0]
        item = Item(sku=sku, name=str(row["Name"]), description=str(row['Description']), price=int(row['Price']))
        item.quantity = quantity
        self.shopping_cart.append(item)
        store.stock.at[sku-1, cnf.QUANTITY] -= quantity
        print("Added to cart")

    @clear_cart
    def buy(self, store: Store):
        check_out = Transaction.from_parent(self)
        for i in self.shopping_cart:
            record = {
                cnf.CUSTOMER_NAME: check_out.name,
                cnf.TX_DT: check_out.ts,
                cnf.CUSTOMER_AGE: check_out.age,
                cnf.LOCATION: check_out.location,
                cnf.DISCOUNT: check_out.discount,
                cnf.SKU_ID: i.sku,
                cnf.QUANTITY: i.quantity,
                cnf.ITEM_PRICE: i.price,
                cnf.ITEM_NAME: i.name,
                cnf.ITEM_DESCRIPTION: i.description,
                "Amount Spent in USD": i.quantity*i.price
            }
            store.log.append(record)
        return check_out


class Transaction(Customer):
    @classmethod
    def from_parent(cls, parent):
        return cls(parent.name, parent.age, parent.location, parent.shopping_cart)

    def __init__(self, name, age, location, shopping_cart, discount: float = 0.0):
        super(Transaction, self).__init__(name, age, location)
        if type(discount) == int:
            discount = float(discount)/100
        self.ts = randomtimestamp(start_year=2010).strftime(
            '%m/%d/%Y %-I:%M:%S %p')  # ts means timestamp
        self.list = shopping_cart
        self.num_items = sum([i.quantity for i in shopping_cart])
        self.discount = discount
        self.total = sum(
            [i.price*i.quantity for i in shopping_cart])*(1-discount)

    def __str__(self):
        return f'Date and time is {self.ts}, {self.num_items} items, Discount is {self.discount}%'
