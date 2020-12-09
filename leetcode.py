#
# def calculate(s: str) -> int:
#     s = s.replace(' ', '')
#     s = '(' + s + ')'
#     def cal_part(s, idx=0):
#         res = 0
#         str_buff = ''
#         cal_flag = 1
#         while idx <= len(s) - 1:
#             if s[idx] == '(':
#                 if cal_flag == 1:
#                     if str_buff != '':
#                         res += int(str_buff)
#                         str_buff = ''
#                     pres, new_idx = cal_part(s, idx + 1)
#                     res += pres
#                     idx = new_idx
#                 elif cal_flag == 0:
#                     if str_buff != '':
#                         res -= int(str_buff)
#                         str_buff = ''
#                     pres, new_idx = cal_part(s, idx + 1)
#                     res -= pres
#                     idx = new_idx
#             elif s[idx] == '+':
#                 if str_buff != '':
#                     if cal_flag == 1:
#                         res += int(str_buff)
#                     elif cal_flag == 0:
#                         res -= int(str_buff)
#                 cal_flag = 1
#                 str_buff = ''
#             elif s[idx] == '-':
#                 if str_buff != '':
#                     if cal_flag == 1:
#                         res += int(str_buff)
#                     elif cal_flag == 0:
#                         res -= int(str_buff)
#                 cal_flag = 0
#                 str_buff = ''
#             elif s[idx] == ')':
#                 if str_buff != '':
#                     if cal_flag == 1:
#                         res += int(str_buff)
#                     elif cal_flag == 0:
#                         res -= int(str_buff)
#                 return res, idx
#             else:
#                 str_buff += s[idx]
#             idx += 1
#         return res
#
#     return cal_part(s)
#
#
#
# a = calculate("(3-(1+(4+5+2)-(3+4)+3+(3+8))-(6+8))")
# b =eval("(3-(1+(4+5+2)-(3+4)+3+(3+8))-(6+8))")
# print(a == b)


def calculate(s: str) -> int:
    s = s.replace(' ', '')
    s = s + '+0'
    def cal_part(idx=0):
        res = 0
        part = 0
        str_buff = ''
        mul_flag = 0
        cal_flag = 1
        while idx <= len(s) - 1:
            if s[idx] == '+':
                if mul_flag == 1:
                    if cal_flag == 1:
                        res += part * int(str_buff)
                    elif cal_flag == 0:
                        res -= part * int(str_buff)
                    str_buff = ''
                    part = 0
                    mul_flag = 0
                elif mul_flag == 2:
                    if cal_flag == 1:
                        res += part // int(str_buff)
                    elif cal_flag == 0:
                        res -= part // int(str_buff)
                    str_buff = ''
                    part = 0
                    mul_flag = 0
                if str_buff != '':
                    if cal_flag == 1:
                        res += int(str_buff)
                    elif cal_flag == 0:
                        res -= int(str_buff)
                cal_flag = 1
                str_buff = ''
            elif s[idx] == '-':
                if mul_flag == 1:
                    if cal_flag == 1:
                        res += part * int(str_buff)
                    elif cal_flag == 0:
                        res -= part * int(str_buff)
                    str_buff = ''
                    part = 0
                    mul_flag = 0
                elif mul_flag == 2:
                    if cal_flag == 1:
                        res += part // int(str_buff)
                    elif cal_flag == 0:
                        res -= part // int(str_buff)
                    str_buff = ''
                    part = 0
                    mul_flag = 0
                if str_buff != '':
                    if cal_flag == 1:
                        res += int(str_buff)
                    elif cal_flag == 0:
                        res -= int(str_buff)
                cal_flag = 0
                str_buff = ''
            elif s[idx] == '*':
                if mul_flag == 1:
                    part = part * int(str_buff)
                    str_buff = ''
                elif mul_flag == 2:
                    part = part // int(str_buff)
                    str_buff = ''
                else:
                    part = int(str_buff)
                    str_buff = ''
                mul_flag = 1
            elif s[idx] == '/':
                if mul_flag == 1:
                    part = part * int(str_buff)
                    str_buff = ''
                elif mul_flag == 2:
                    part = part // int(str_buff)
                    str_buff = ''
                else:
                    part = int(str_buff)
                    str_buff = ''
                mul_flag = 2
            else:
                str_buff += s[idx]
            idx += 1
        return res
    return cal_part()

a = calculate("2*3-4")
b =eval("3+2*2")