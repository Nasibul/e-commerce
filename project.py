import datetime

class Customer:
    '''
    This is a live autonomous agent - it has its own random control
    '''
    def __init__(self, name: str, age: int, location: str):
        #super(Customer, self).__init__()
        assert type(name) == str, f"Name must be a string, instead found {name}"
        self.name = name
        self.age = age
        self.location = location

    def buy(self, item):
        instance = Transaction(datetime.datetime.now(), 13)
        print("hello", instance.tx_datetime)

class Item:
    '''
    This is controlled by the store
    '''
    def __init__(self, sku: float, name: str, description: str, price: float):
        self.sku = sku
        self.name = name
        self.description = description
        self.price = price

class Transaction(Customer, Item):
    '''
    This is controlled by the customer and the store, but it needs to inherit the item too
    '''
    def __init__(self, datetime: datetime, value: float, discount: float=None):
        self.tx_datetime = datetime
        self.value = value
        self.discount = discount

class Store:
    '''
    Controls - spawns items, approves transaction, manages inventory, deal with returns
    '''
    def __init__(self, location: str, items: str):
        self.location = location
        self.items = items

dummy = Customer("Nash", 24, "NYC")
dummyitem = Item(234234, "ice cream", 'chocolate', 5)
#dummytransaction = Transaction(datetime.datetime.now(), 13, "alex", 13, "nyc")
#print(dummytransaction.name)
dummy.buy(item=dummyitem)

'''
- store to spawn items
- store to have a stock count property per item category
'''