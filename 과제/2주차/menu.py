from dataclasses import dataclass

@dataclass
class Menu:
    name: str
    price: int

@dataclass
class MenuBook:
    drink:list[Menu]
    desert:list[Menu]

    drinks = [{"menu" :"아메리카노", "price": 2000}, {"menu" :"카페라떼", "price":2500}, {"menu" :"아이스티", "price":3000}]
    deserts = [{"menu" :"소금빵", "price": 3000}, {"menu" :"베이글", "price":4000}, {"menu" :"초콜릿 케잌", "price":5000}]

    def __init__(self):
        self.drink = []
        drinkIter = iter(self.drinks)
        while True :
            try:
                menu = next(drinkIter)
                self.drink.append(Menu(menu["menu"], menu["price"]))
            except StopIteration : break

        self.desert = []
        desertIter = iter(self.deserts)
        while True :
            try:
                menu = next(desertIter)
                self.desert.append(Menu(menu["menu"], menu["price"]))
            except StopIteration : break