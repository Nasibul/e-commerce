import datetime
import pandas as pd
import copy


def clear_cart(func):
    def wrapper(*args):
        func(*args)
        args[0].shopping_cart = []
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
        assert type(
            location) == str, f"Location must be a string, instead found {location}"
        self.location = location
        self.stock = []

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
log = pd.DataFrame(columns=["Customer Name", "Transaction Date and Time", "Customer Age",
                            "Cart", "Number of Items", "Discount", "Total"])


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

    def grab(self, item: Item, store: Store, quantity: int = 1):
        # this method is used to take an item and input into a transaction object
        if item in store.stock:
            index = store.stock.index(item)
            if store.stock[index].quantity < quantity:
                print(
                    f"Quantity too high. We only have {store.stock[index].quantity} of this item.")
            else:
                item.quantity -= quantity
                new_item = copy.deepcopy(item)
                skus = [i.sku for i in self.shopping_cart]
                if new_item.sku in skus:
                    self.shopping_cart[skus.index(
                        new_item.sku)].quantity += quantity
                else:
                    new_item.quantity = quantity
                    self.shopping_cart.append(new_item)
        else:
            print('Item not in stock')
        print('SHOPPING CART')
        print(*self.shopping_cart, sep='\n')
        print('\n')

    @clear_cart
    def buy(self):
        check_out = Transaction.from_parent(self)
        print(check_out.name)
        print(check_out)
        print(*check_out.list, sep='\n')
        print(f'Total is ${"{:.2f}".format(check_out.total)}')
        receipt = {
            "Customer Name": check_out.name,
            "Transaction Date and Time": check_out.ts,
            "Customer Age": check_out.age,
            "Cart": check_out.list,
            "Number of Items": check_out.num_items,
            "Discount": check_out.discount,
            "Total": check_out.total
        }
        global log
        log = pd.concat([log, pd.DataFrame(receipt)], ignore_index=True)
        return None

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
