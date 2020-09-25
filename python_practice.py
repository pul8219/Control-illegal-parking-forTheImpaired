
#for line in range(1,11):
#    print(" "*(10 - line) + "*" * line)


# 1부터 100사이의 숫자 중 홀수의 합이 1000이 넘지 않는 위치가 어디인지를 묻는 프로그램

# sum = 0
#
# for n in range(1, 101, 2):
#     sum += n
#     if(sum > 1000):
#         break
#
# print("1 + 3 + 5 + … + ", n-2, "=", sum - n)


#
# sum = 0
# n = 1
# while(n < 101):
#     sum += n
#     if sum > 1000:
#         break
#     n = n + 2
#
# print("1 + 3 + 5 + … + ", n-2, "=", sum - n)



#random.randint(a,b) a,b 사이의 랜덤한 정수 반환. a,b도 범위에 포함

#1~100 사이의 무작위 정수 10개를 갖는 리스트를 생성하는 프로그램
# import random
#
# new_list = []
# for i in range(10): #range(10) 0부터 10미만의 숫자 범위
#     new_list.append(random.randint(1,100))
#     #new_list.insert(i, random.randint(1,100))
#new_list[i] = random.randint(1,100) #new_list가 비어있기 때문에 오류남
#
# print(new_list)


import random
new_list = []
i = 0
while i < 10:
    new_list.append(random.randint(1,100))
    #new_list.insert(i, random.randint(1,100))
    i = i + 1
print(new_list)
