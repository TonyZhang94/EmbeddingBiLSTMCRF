
ch = "發"
if u'\u4e00' <= ch <= u'\u9FA5':
    print("in")

print(0x9fff - 0x4e00)
print(0x9fbf - 0x4e00)
print(0xA000 - 0x4e00)
print(0xA008 - 0x4e00)

print(u'\u9FA5')
print(u'\u9FA7')
print(u'\u9FD5')  # 最后一个可打印汉字

# 简繁体转换存在一对一，一对多，多对一，多对多的问题，此处不做任何转换操作

print("我".encode("unicode_escape"))
temp = "我".encode("unicode_escape")
print(temp)
temp = str(temp).replace("b'\\", "").replace("'", "")
print(temp)
print("\u6211")

print(type('\u9FD5'))
print(type(u'\u9FD5'))


digits = [str(n) for n in range(10)] + ['A', 'B', 'C', 'D', 'E', 'F']
digits2num = {digits[num]: num for num in range(16)}
# print(digits)
# for p1 in digits:
#     for p2 in digits:
#         for p3 in digits:
#             for p4 in digits:
#                 ch = u"%s%s%s%s" % (p1, p2, p3, p4)
#                 if u'\u4e00' <= ch <= u'\u9FA5':
#                     print(temp)
#                     exit()

ch = "我"
u_form = ch.encode("unicode_escape")
u_form = str(u_form).replace("b'\\\\u", "").replace("'", "")
print(u_form)
rank = 0
for x in list(u_form):
    rank = rank * 16 + digits2num[x]
print('0x%x' % rank)

print(u'\u4e00')
print(u'\u9FFF')
