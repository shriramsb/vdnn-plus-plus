import matplotlib.pyplot as plt
import sys

f = open(sys.argv[1], "r")

avg_total = 0.0
flag = 0
count = 0
total_time_list = []
x = []
i = 1
while True:
	cur_total = 0.0
	while True:
		cur_line = f.readline()
		cur_line_split = cur_line.split()
		if (cur_line == ""):
			flag = 1
			break
		if (cur_line[0] == '\n'):
			break
		cur_total += float(cur_line_split[1])

	avg_total += cur_total
	count += 1
	total_time_list.append(cur_total)
	x.append(i)
	i += 1

	if (flag == 1):
		break

count -= 1
del x[-1]
del total_time_list[-1]

avg_total /= count
print ("avg_total: ", avg_total)
plt.scatter(x, total_time_list)
plt.show()