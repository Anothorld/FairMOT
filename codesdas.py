# import sys
# for line in sys.stdin:
#     a = line.split()
#     print(int(a[0]) + int(a[1]))
#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
# import sys
# if __name__ == "__main__":
#     # 读取第一行的n
#     n = int(sys.stdin.readline().strip())
#     ans = 0
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         values = list(map(int, line.split()))
#         for v in values:
#             ans += v
#     print(ans)

# def find_value(l):
#     res = l[-1]
#     for i in range(len(l) - 1, 0, -1):
#         if res >= 0:
#             res -= l[i - 1]
#         else:
#             res += l[i - 1]
#     if res == 0:
#         return 1
#     else:
#         return 0

# T = int(input())
# for i in range(T):
#     n = int(input())
#     l = map(int, input().split(' '))
#     l = sorted(l)
#     res = 0
#     last = 0
#     for j in range(n):
#         flag = find_value(l[j:])
#         if flag == 1:
#             last = j
#             break
#     res = sum(l[:last])
#     print(res)

# T = int(input())
# for i in range(1):
#     n = 5
#     l = [30, 60, 5, 15, 30]
#     l = sorted(l)
#     res = 0
#     last = 0
#     for j in range(n):
#         flag = find_value(l[j:])
#         if flag == 1:
#             last = j
#             break
#     res = sum(l[:last])
#     print(res)


# T = int(input())
# T = 2
# n_l = [2, 1]
# l_l = [[20, 25], [8]]
# t_l = [[40], []]
#
# # for i in range(T):
# #     n_l.append(int(input()))
# #     l_l.append(list(map(int, input().strip().split(' '))))
# #     t_l.append(list(map(int, input().strip().split(' '))))
# for i in range(T):
#     res = []
#     n = n_l[i]
#     l = l_l[i]
#     t = t_l[i]
#     for idx in range(len(l)):
#         if idx == 0:
#             res.append(l[idx])
#         elif idx == 1:
#             res.append(min(res[idx - 1] + l[idx], t[idx - 1]))
#         else:
#             res.append(min(res[idx - 1] + l[idx], t[idx - 1] + res[idx - 2]))
#
#     all_mint = res[-1] // 60
#     sec = res[-1] % 60
#     hour = all_mint // 60
#     mint = all_mint % 60
#     print('{:02}:{:02}:{:02}{}'.format((8 + hour) % 12, mint, sec, 'am' if (8 + hour) // 12 == 0 else 'pm'))
# T = 1
# n_l = [5]
# l_l = [[30, 60, 5, 15, 30]]

# def find_value(l):
#     res = l[-1]
#     for i in range(len(l) - 1, 0, -1):
#         if res >= 0:
#             res -= l[i - 1]
#         else:
#             res += l[i - 1]
#     if res == 0:
#         return 1
#     else:
#         return 0
#
#
# for i in range(T):
#     n = n_l[i]
#     l = sorted(l_l[i])
#     res = 0
#     last = 0
#     for j in range(n):
#         flag = find_value(l[j:])
#         if flag == 1:
#             last = j
#             break
#     res = sum(l[:last])
#     print(res)
M, N, P, Q = [int(_) for _ in input().split()]
data = []
for i in range(M):
    buff = [_ for _ in input()]
    data.append(buff)

res = [[0 for _ in range(N)] for _ in range(M)]

for i in range(M):
    for j in range(N):
        k = [[data[i][j] for _ in range(Q)] for _ in range(P)]
        buff = 0
        for x in range(-P // 2, P // 2 + 1):
            for y in range(-Q // 2, Q // 2 + 1):
                if i + x < 0 or i + x >= len(data) or j + y < 0 or j + y >= len(data[0]):
                    continue
                else:
                    if data[i + x][j + y] == data[i][j]:
                        buff += 1
        res[i][j] = res

for i in range(M):
    print(''.join(list(map(str, M[i]))))