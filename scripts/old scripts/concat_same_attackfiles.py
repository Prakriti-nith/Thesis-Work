attack_files = []

adduser = []
hydra_ftp = []
hydra_ssh = []
java = []
meterpreter = []
web_shell = []

output = ['../Attack_Data_Master/Training_attacks/adduser_train.txt', '../Attack_Data_Master/Training_attacks/hydra_ftp_train.txt', '../Attack_Data_Master/Training_attacks/hydra_ssh_train.txt',
'../Attack_Data_Master/Training_attacks/java_meterpreter_train.txt', '../Attack_Data_Master/Training_attacks/meterpreter_train.txt', '../Attack_Data_Master/Training_attacks/web_shell_train.txt']

for i in range(1,6):
    adduser.append('../Attack_Data_Master/Adduser_' + str(i) + '/outfile.txt')
    hydra_ftp.append('../Attack_Data_Master/Hydra_FTP_' + str(i) + '/outfile.txt')
    hydra_ssh.append('../Attack_Data_Master/Hydra_SSH_' + str(i) + '/outfile.txt')
    java.append('../Attack_Data_Master/Java_Meterpreter_' + str(i) + '/outfile.txt')
    meterpreter.append('../Attack_Data_Master/Meterpreter_' + str(i) + '/outfile.txt')
    web_shell.append('../Attack_Data_Master/Web_Shell_' + str(i) + '/outfile.txt')

attack_files.append(adduser)
attack_files.append(hydra_ftp)
attack_files.append(hydra_ssh)
attack_files.append(java)
attack_files.append(meterpreter)
attack_files.append(web_shell)

for i in range(len(output)):
    with open(output[i], 'wb+') as outfile:
        for filename in attack_files[i]:
            with open(filename, 'rb') as readfile:
                for line in readfile:
                    outfile.write(line)
