import datetime

class Customer:
    def __init__(self, name, age, location):
        self.name = name
        self.age = age
        self.location = location
    def buy(self, ):
        Transaction.value += Item.price


class Transaction:
    def __init__(self, datetime, value, discount):
        self.datetime = datetime
        self.value = value
        self.discount = discount
    

class Item:
    def __init__(self, sku, name, description, price):
        self.sku = sku
        self.name = name
        self.description = description
        self.price = price

class Store:
    def __init__(self, location, items):
        self.location = location
        self.items = items

