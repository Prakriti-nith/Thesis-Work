import os
import glob

outfilename = []
for i in range(1,6):
    outfilename.append('../Attack_Data_Master/Adduser_' + str(i) + '/outfile.txt')
    outfilename.append('../Attack_Data_Master/Hydra_FTP_' + str(i) + '/outfile.txt')
    outfilename.append('../Attack_Data_Master/Hydra_SSH_' + str(i) + '/outfile.txt')
    outfilename.append('../Attack_Data_Master/Java_Meterpreter_' + str(i) + '/outfile.txt')
    outfilename.append('../Attack_Data_Master/Meterpreter_' + str(i) + '/outfile.txt')
    outfilename.append('../Attack_Data_Master/Web_Shell_' + str(i) + '/outfile.txt')

for f in outfilename:
    os.remove(f)
    