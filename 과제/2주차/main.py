# 1. 기존 CLI 프로그램(동기)을 아래 조건에 맞춰 리팩토링해 보세요.
# O 1-1. 모듈/패키지 분리 – 최소 2개 이상의 파일로 구조화 (main.py, utils.py 등)
# O 1-2. 타입 힌트 – 모든 함수에 타입 힌트 명시
# O 1-3. 예외 처리 – 잘못된 입력에 대해 try/except로 안정성 보장
# O 1-4. 이터레이터/제네레이터 활용 – 데이터 순차 처리 기능에 활용 (예: 로그 기록 출력) - 메뉴 불러오기
# 1-5. 비동기 처리 – asyncio를 사용해 병렬 처리 적용 - 주문 타이머

from order import Order, DesertOrder, DrinkOrder
from utils import getUserSelect
import asyncio

async def mainProc():
    orders:list[Order] = []

    while True :
        
        category = await getUserSelect("원하시는 기능을 선택해주세요.", ["주문하기", "장바구니", "결제하기", "종료하기"])

        match category:
            case "주문하기" :
                pick = await getUserSelect("메뉴 종류를 선택하세요", ["음료", "디저트"])
                if pick == "음료" :
                    order = DrinkOrder()
                    orders.append(order)
                    await order.main()
                elif pick == "디저트" :
                    order = DesertOrder()
                    orders.append(order)
                    await order.main()
                
            case "장바구니" :
                print("###########  장바구니  ###########")
                totalPrice = 0
                for i, order in enumerate(orders):
                    print(f"{i+1}. 메뉴 : {order.menuName} {order.optionToStr()} ,가격 : {order.price}")
                    totalPrice += (order.price)

                print(f"총 금액 : {totalPrice}")
                print("##################################")
                print()

            case "결제하기" :
                pick = await getUserSelect("결제 수단을 선택해주세요", ["신용카드", "카카오 페이", "현금"])

                print("###########  영수증  ###########")
                totalPrice = 0

                for i, order in enumerate(orders):
                    print(f"{i+1}. 메뉴 : {order.menuName} {order.optionToStr()} ,가격 : {order.price}")
                    totalPrice += (order.price)

                print(f"결제금액 : {totalPrice}")
                print(f"결제수단 : {pick}")
                print("################################")
                print("주문을 종료합니다.")
                return

            case "종료하기":
                print("주문을 종료합니다.")
                return
            case _:
                print("다시 선택해주세요")
                
async def timeProc():
    while True : 
        await asyncio.sleep(5)
        print("광고 : 안녕하세요 ~~~~를 추천 드립니다!!!!")

async def main():
    await asyncio.gather(
        mainProc(),
        timeProc()
    )

asyncio.run(main())
