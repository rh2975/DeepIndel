
file_name = "HG002_SVs_Tier1_noVDJorXorY_v0.6.2.bed"


file = open(file_name, "r")
write_file = open("del_exp.bed", "w")
lines = file.readlines()

for i in range(len(lines)):
    # print(lines[i])
    # print(lines[i].split("\t")[2])
    start = int(lines[i].split("\t")[1])
    end = int(lines[i].split("\t")[2])
    # print(start,end)
   
    write_file.write("chr"+lines[i].split("\t")[0]+"\t" +
                     str(start)+"\t"+str(end)+"\n")


