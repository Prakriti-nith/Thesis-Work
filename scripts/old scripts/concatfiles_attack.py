import shutil
from glob import glob

outfilename = []
files = []
for i in range(1,6):
    outfilename.append('../Attack_Data_Master/Adduser_' + str(i) + '/outfile.txt')
    files.append(glob('../Attack_Data_Master/Adduser_' + str(i) + '/*.txt'))
    outfilename.append('../Attack_Data_Master/Hydra_FTP_' + str(i) + '/outfile.txt')
    files.append(glob('../Attack_Data_Master/Hydra_FTP_' + str(i) + '/*.txt'))
    outfilename.append('../Attack_Data_Master/Hydra_SSH_' + str(i) + '/outfile.txt')
    files.append(glob('../Attack_Data_Master/Hydra_SSH_' + str(i) + '/*.txt'))
    outfilename.append('../Attack_Data_Master/Java_Meterpreter_' + str(i) + '/outfile.txt')
    files.append(glob('../Attack_Data_Master/Java_Meterpreter_' + str(i) + '/*.txt'))
    outfilename.append('../Attack_Data_Master/Meterpreter_' + str(i) + '/outfile.txt')
    files.append(glob('../Attack_Data_Master/Meterpreter_' + str(i) + '/*.txt'))
    outfilename.append('../Attack_Data_Master/Web_Shell_' + str(i) + '/outfile.txt')
    files.append(glob('../Attack_Data_Master/Web_Shell_' + str(i) + '/*.txt'))

for i in range(len(outfilename)):
    with open(outfilename[i], 'wb+') as outfile:
        for filename in files[i]:
            if filename == outfilename[i]:
                # don't want to copy the output into the output
                continue
            with open(filename, 'rb') as readfile:
                for line in readfile:
                    outfile.write(line)
                    