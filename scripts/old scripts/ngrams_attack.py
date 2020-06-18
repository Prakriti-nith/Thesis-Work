n = 2

attack_files = ['../Attack_Data_Master/Training_attacks/adduser_train.txt', '../Attack_Data_Master/Training_attacks/hydra_ftp_train.txt', '../Attack_Data_Master/Training_attacks/hydra_ssh_train.txt',
'../Attack_Data_Master/Training_attacks/java_meterpreter_train.txt', '../Attack_Data_Master/Training_attacks/meterpreter_train.txt', '../Attack_Data_Master/Training_attacks/web_shell_train.txt']

all_words = []
for attack_file in attack_files:
    temp = []
    with open(attack_file, "r") as f:
        temp = f.read().split()
    all_words.append(temp)

freq_arr = []

for train_calls in all_words:
    dic = {}
    for i in range(len(train_calls)-n+1):
        temp = ""
        for j in range(i,i+n):
            temp = temp + str(train_calls[j]) + ", "
        if temp in dic:
            dic[temp] = dic[temp] + 1
        else:
            dic[temp] = 1
    freq_arr.append(dic)

output_files = ['../Attack_Data_Master/Training_attacks/adduser_ngram_train.txt', '../Attack_Data_Master/Training_attacks/hydra_ftp_ngram_train.txt', '../Attack_Data_Master/Training_attacks/hydra_ssh_ngram_train.txt',
'../Attack_Data_Master/Training_attacks/java_meterpreter_ngram_train.txt', '../Attack_Data_Master/Training_attacks/meterpreter_ngram_train.txt', '../Attack_Data_Master/Training_attacks/web_shell_ngram_train.txt']

for i in range(len(output_files)):
    f= open(output_files[i],"w+")
    print("\n\n")
    for x, y in freq_arr[i].items():
      print(x, y)
      f.write("%s%d\n" % (x, y))
