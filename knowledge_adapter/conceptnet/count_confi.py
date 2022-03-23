# know_num = 0
#
# dic = {}
# for line in open("conceptnet-assertions-5.7.0_merge.csv", "r"):
#     know_num += 1
#     relation = line.split("\t")[0]
#     if dic.get(relation) is None:
#         dic[relation] = 0
#     else:
#         dic[relation] += 1
#
# print(dic)
# print("--------------------")
# print(know_num)

know_num = 0
num = 0
# dic = {}
for line in open("conceptnet-assertions-5.7.0_merge.csv", "r"):
    know_num += 1
    confi = float(line.split("\t")[-1])
    if confi > 2.0:
        num +=1

print(num)
print("--------------------")
print(know_num)