import asyncio

async def getUserSelect(comment:str, decisions:list[str])->str:
    print(comment)
    for i, decision in enumerate(decisions) : print(f"{i+1}, {decision}")

    while True :
        try:
            user_input = await asyncio.to_thread(input, "")
            pick = int(user_input)
            if pick < 1 or len(decisions) < pick  : raise Exception
            return decisions[pick-1]
        except :
            print("잘못된 입력입니다. 다시 선택해주세요")