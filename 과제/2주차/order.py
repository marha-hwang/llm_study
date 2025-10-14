from abc import ABC, abstractmethod
import asyncio
from utils import getUserSelect
from menu import Menu, MenuBook


class Order(ABC):
    menuName:str
    price:int
    options:list[str]

    menuBook = MenuBook()

    def __init__(self):
        self.menuName: str = ""
        self.price: int = 0
        self.options: list[str] = []

    @abstractmethod
    def order(self):
        pass

    # 주문진행 프로세스
    async def main(self):
        print("메뉴를 선택하세요")
        await self.order()
        print("메뉴를 추가하였습니다.")

    # 옵션 출력함수
    def optionToStr(self)->str :
        return ", ".join(self.options)

class DesertOrder(Order):
    async def order(self):
        for i, desert in enumerate(self.menuBook.desert) :
            print(f"{i+1}. 메뉴명 : {desert.name}, 가격 : {desert.price}")

        pick = int(await asyncio.to_thread(input, ""))
        self.menuName = self.menuBook.desert[pick-1].name
        self.price = self.menuBook.desert[pick-1].price

class DrinkOrder(Order):
    async def order(self):
        for i, drink in enumerate(self.menuBook.drink) :
             print(f"{i+1}. 메뉴명 : {drink.name}, 가격 : {drink.price}")

        pick = int(await asyncio.to_thread(input, ""))
        self.menuName = self.menuBook.drink[pick-1].name
        self.price = self.menuBook.drink[pick-1].price

        option1 = await getUserSelect("샷추가 여부를 선택해주세요(+500)", ["y", "n"])
        if option1 == "y" :
            self.options.append("샷 추가")
            self.price += 500
        print()

        option2 = await getUserSelect("ice/hot을 선택해주세요", ["ice", "hot"])
        self.options.append(option2)
        print()

        option3 = await getUserSelect("size선택 : S/M(+500)/L(+1000)", ["S", "M", "L"])
        self.options.append(option3)
        if option3 == "M" : self.price += 500
        if option3 == "L" : self.price += 1000
