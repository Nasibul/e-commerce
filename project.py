import datetime
import math 
'''
Will have to build a items df for all of the items.
Same for customers and stores. 
Can utilize the python random package to generate numbers that simulate customers
'''
def clear_cart(func):
    def wrapper(*args):
        func(*args)
        args[0].shopping_cart = []
    return wrapper

class Item:
    '''
    This is controlled by the store
    '''
    def __init__(self, sku: float, name: str, description: str, price: float, quantity: int=1):
        if type(price) == int: 
            price = float(price)
        assert type(sku) == int, f"SKU must be a float, instead found {sku}"
        assert type(name) == str, f"Name must be a string, instead found {name}"
        assert type(description) == str, f"description must be a string, instead found {description}"
        assert type(price) == float, f"Price must be a float, instead found {price}"
        self.sku = sku
        self.name = name.title()
        self.description = description
        self.price = round(price, 2)
        self.quantity = quantity

    def __str__(self):
        return f'SKU#{self.sku}, {self.name}, {self.description}, ${"{:.2f}".format(self.price)}, Quantity= {self.quantity}'

class Customer:
    '''
    This is a live autonomous agent - it has its own random control
    '''
    def __init__(self, name: str, age: int, location: str):
        #super(Customer, self).__init__()
        assert type(name) == str, f"Name must be a string, instead found {name}"
        assert type(age) == int, f"Age must be an integer, instead found {age}"
        assert type(location) == str, f"Location must be a string, instead found {location}"
        self.name = name
        self.age = age
        self.location = location
        self.shopping_cart = []

    def __str__(self):
        return f'{self.name}, {self.age}, {self.location}'

    def grab(self, item: Item, quantity: int=1):
        #this method is used to take an item and input into a transaction object
        item.quantity = quantity
        self.shopping_cart.append(item)
    
    @clear_cart
    def buy(self):
        '''
        This method has to create a transaction and check with the store class
        Store class has to have enough of the item and then the appropriate amount is deducted
        '''
        check_out = Transaction.from_parent(self)
        print(check_out.name)
        print(check_out)
        print(*check_out.list, sep='\n')
        print(f'Total is ${"{:.2f}".format(check_out.total)}')

    def return_item(self, item: Item):
        pass

class Transaction(Customer):
    '''
    This is controlled by the customer and the store, but it needs to inherit the item too
    It needs to be able to have multiple different items and multiple items of the 
    same kind. For example, 5 packs of candy, 2 bags of chips, 3 bundles of paper plates.
    Should not be called by the user but only by the customer class
    '''
    @classmethod
    def from_parent(cls, parent):
        return cls(parent.name, parent.age, parent.location, parent.shopping_cart)
    
    def __init__(self, name, age, location, shopping_cart, ts: datetime=datetime.datetime.now(), discount: float=0.0):
        super(Transaction, self).__init__(name, age, location)
        if type(discount) == int: 
            discount = float(discount)/100
        self.ts = ts.strftime('%m/%d/%Y %H:%M:%S %p') #ts means timestamp
        self.list = shopping_cart
        self.num_items = sum([i.quantity for i in shopping_cart])
        self.discount = discount
        self.total = sum([i.price*i.quantity for i in shopping_cart])*(1-discount)
    
    def __str__(self):
        return f'Date and time is {self.ts}, {self.num_items} items, Discount is {self.discount}'

class Store:
    '''
    Controls - spawns items, approves transaction, manages inventory, deal with returns
    '''
    def __init__(self, location: str):
        assert type(location) == str, f"Location must be a string, instead found {location}"
        self.location = location 
    
    def restock(item, quantity):
        pass

dummy = Customer("Nash", 24, "NYC")
dummyitem = Item(234234, "ice cream", 'Chocolate', 6)
dummyitem2 = Item(3242443, "ice cream", 'Vanilla', 5)
dummyitem3 = Item(3222443, "ice cream", 'Strawberry', 6)
#dummytransaction = Transaction(datetime.datetime.now(), 13, "alex", 13, "nyc")
#print(dummytransaction.name)
dummy.grab(dummyitem, 3)
dummy.grab(dummyitem2, 2)
dummy.grab(dummyitem3, 1)
dummy.buy()
print(dummy.shopping_cart)
'''
- store to spawn items
- store to have a stock count property per item category
- store to have items depleted 
- customers need to buy and return
'''