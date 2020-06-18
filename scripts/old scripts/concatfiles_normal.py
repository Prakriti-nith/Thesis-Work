filenames = []

for x in range(1,834):
    if x<10:
        filenames.append("../Training_Data_Master/UTD-000" + str(x) + ".txt")
    elif x<100:
        filenames.append("../Training_Data_Master/UTD-00" + str(x) + ".txt")
    else:
        filenames.append("../Training_Data_Master/UTD-0" + str(x) + ".txt")

with open('../Training_Data_Master/outfile.txt', 'w') as outfile:
    for fname in filenames:
        if fname == '../Training_Data_Master/outfile.txt':
            # don't want to copy the output into the output
            continue
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
