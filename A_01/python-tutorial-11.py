"""
Modules & Packages
"""

import sales.pricing as pricing
import sales.product as product

net_price = pricing.get_net_price(
    price=100,
    tax_rate=0.01
)

print(net_price)

tax = pricing.get_tax(100)
print(tax)

tax = product.get_tax(100)
print(tax)