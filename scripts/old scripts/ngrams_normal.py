n = 2

all_words = []
with open('../Training_Data_Master/outfile.txt', "r") as f:
    all_words = f.read().split()
dic = {}

for i in range(len(all_words)-n+1):
	temp = ""
	for j in range(i,i+n):
		temp = temp + str(all_words[j]) + ", "
	if temp in dic:
		dic[temp] = dic[temp] + 1
	else:
		dic[temp] = 1

f= open("../Training_Data_Master/ngrams_train.txt","w+")
	
for x, y in dic.items():
  print(x, y)
  f.write("%s%d\n" % (x, y))
