# pricing.py

def get_net_price(price, tax_rate, discount=0):
    discounted_price = price * (1 - discount) 
    net_price = discounted_price * (1 + tax_rate) 
    return net_price


def get_tax(price, tax_rate=0):
    return price * tax_rate
