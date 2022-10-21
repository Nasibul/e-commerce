import datetime
from unittest import result
import pandas as pd
import copy
import config as cnf

def clear_cart(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        args[0].shopping_cart = []
        return result
    return wrapper

def validate_stock(func):
    def wrapper(*args, **kwargs):
        item = kwargs['item']
        store = kwargs['store']
        quantity = kwargs['quantity']
        error = None
        if item in store.stock:
            index = store.stock.index(item)
            if store.stock[item].quantity < quantity:
                    error = f"Quantity too high. We only have {store.stock[index].quantity} of this item."
            else:
                return func(*args, **kwargs)
        else:
            error = 'Item not in stock'
        return error
    return wrapper

class Item:
    def __init__(self, sku: int, name: str, description: str, price: float, quantity: int = 1):
        if type(price) == int:
            price = float(price)
        assert type(sku) == int, f"SKU must be a int, instead found {sku}"
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
        self.quantity = quantity

    def __str__(self):
        return f'SKU#{self.sku}, {self.name}, {self.description}, ${"{:.2f}".format(self.price)}, Quantity= {self.quantity}'


class Store:
    def __init__(self, location: str):
        assert type(location) == str, f"Location must be a string, instead found {location}"
        self.location = location
        self.stock = []
        self.log = pd.DataFrame(columns=[cnf.CUSTOMER_NAME, cnf.TX_DT, \
                                         cnf.CUSTOMER_AGE, cnf.CART, cnf.NUMBER_OF_ITEMS, \
                                         cnf.DISCOUNT, cnf.TOTAL])

    def restock(self, item, quantity):
        item.quantity = quantity
        self.stock.append(item)

    def display_stock(self):
        print('STOCK')
        print(*self.stock, sep='\n')
        print('\n')

    def __str__(self):
        return f'{self.location} Store'


# cool_store = Store("NYC")

class Customer:
    def __init__(self, name: str, age: int, location: str):
        #super(Customer, self).__init__()
        assert type(
            name) == str, f"Name must be a string, instead found {name}"
        assert type(age) == int, f"Age must be an integer, instead found {age}"
        assert type(
            location) == str, f"Location must be a string, instead found {location}"
        self.name = name
        self.age = age
        self.location = location
        self.shopping_cart = []

    def __str__(self):
        return f'{self.name}, {self.age}, {self.location}'

    @validate_stock
    def grab(self, item: Item, store: Store, quantity: int = 1):
        # this method is used to take an item and input into a transaction object
        item.quantity -= quantity
        new_item = copy.deepcopy(item)
        skus = [i.sku for i in self.shopping_cart]
        if new_item.sku in skus:
            self.shopping_cart[skus.index(
                new_item.sku)].quantity += quantity
        else:
            new_item.quantity = quantity
            self.shopping_cart.append(new_item)
        shopping_cart_contents = f'SHOPPING CART {self.shopping_cart}'
        return shopping_cart_contents

    @clear_cart
    def buy(self, store:Store):
        check_out = Transaction.from_parent(self)
        receipt = {
            cnf.CUSTOMER_NAME: check_out.name,
            cnf.TX_DT: check_out.ts,
            cnf.CUSTOMER_AGE: check_out.age,
            cnf.CART: check_out.list,
            cnf.NUMBER_OF_ITEMS: check_out.num_items,
            cnf.DISCOUNT: check_out.discount,
            cnf.TOTAL: check_out.total
        }
        store.log = pd.concat([store.log, pd.DataFrame(receipt)], axis=0)
        return check_out

    def return_item(self, item: Item, quantity: int, store: Store):
        index = store.stock.index(item)
        store.stock[index].quantity += quantity
        new_item = copy.deepcopy(item)
        new_item.quantity = quantity
        print(f"Item {new_item} returned to store")


class Transaction(Customer):
    @classmethod
    def from_parent(cls, parent):
        return cls(parent.name, parent.age, parent.location, parent.shopping_cart)

    def __init__(self, name, age, location, shopping_cart, ts: datetime = datetime.datetime.now(), discount: float = 0.0):
        super(Transaction, self).__init__(name, age, location)
        if type(discount) == int:
            discount = float(discount)/100
        self.ts = ts.strftime('%m/%d/%Y %-I:%M:%S %p')  # ts means timestamp
        self.list = shopping_cart
        self.num_items = sum([i.quantity for i in shopping_cart])
        self.discount = discount
        self.total = sum(
            [i.price*i.quantity for i in shopping_cart])*(1-discount)

    def __str__(self):
        return f'Date and time is {self.ts}, {self.num_items} items, Discount is {self.discount}%'
