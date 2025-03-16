import h5py
import numpy as np

# insert your file name here
filename = "./pos_Indel/pad_56.hdf5"
# filename = "./neg_nonIndel/pad_50_shift_64.hdf5"

class_type = 1 # 1 for (pos)itive i.e. indels ; 0 for (neg)ative i.e. non-indels


file = open("chromosomes_before_sort.txt", "w")

with h5py.File(filename, "r") as ff:

    level_1 = list(ff.keys())[0]
    print("Top group : ",level_1) #summaries
   
    level_1_files = list(ff[level_1].keys())
    print("total files under sum: ", len(level_1_files))
    for i in range(len(level_1_files)):
            file.write(level_1_files[i]+"\n")

file.close()

chr_order = ['chr1_', 'chr2_', 'chr3_', 'chr4_', 'chr5_', 'chr6_', 'chr7_', 'chr8_', 'chr9_', 'chr10_', 'chr11_', 'chr12_', 'chr13_', 'chr14_', 'chr15_', 'chr16_', 'chr17_', 'chr18_', 'chr19_', 'chr20_', 'chr21_', 'chr22_']

write_file = open("chromosomes_after_sort.txt", "w")
file = open("chromosomes_before_sort.txt", "r")
lines = file.readlines()
final_array = []
count = 0
for i in range(len(chr_order)):
    
    curr_chr = chr_order[i].split("_")[0]
    curr_list_num = []
    curr_list = {}
    for j in range(len(lines)):
        if lines[j].startswith(chr_order[i]):
            
            cc = lines[j].split("_")[1]
            curr_list[cc] = lines[j].split("_")[2][:-1]
            curr_list_num.append(int(cc))
            count += 1
    sorted_list_num = sorted(curr_list_num)
    # for ll in range(len(sorted_list_num)):
    #     print(sorted_list_num[ll])

    # print(sorted_list_num)
    # print(len(curr_list),len(sorted_list_num))

    for j in range(len(sorted_list_num)):
        # print(curr_list[str(sorted_list_num[j])])
        final_array.append(curr_chr + "_" + str(sorted_list_num[j]) + "_" + curr_list[str(sorted_list_num[j])])

for k in range(len(final_array)):
    # print(final_array[k])
    write_file.write(final_array[k] + "\n")

# print(final_array)   
print(count)
print(len(final_array))



write_file = open("data_matrix_with_explantions.txt", "w")
file = open("chromosomes_after_sort.txt", "r")
lines = file.readlines()

# list for collecting all 33 by 26 2d arrays
sample_data = []

with h5py.File(filename, "r") as f:

    level_1 = list(f.keys())[0]
    print("Top group : ",level_1) #summaries
    # write_file.write("Top group : "+level_1+"\n")
    level_1_files = list(f[level_1].keys())
    # print("Files under summaries : ", level_1_files)
    # chr_order = ['chr1_', 'chr2_', 'chr3_', 'chr4_', 'chr5_', 'chr6_', 'chr7_', 'chr8_', 'chr9_', 'chr10_', 'chr11_', 'chr12_', 'chr13_', 'chr14_', 'chr15_', 'chr16_', 'chr17_', 'chr18_', 'chr19_', 'chr20_', 'chr21_', 'chr22_']
    sorted_level_1_files = []
    for i in range(len(lines)):
        # if lines[i][:-1].startswith('chr21_'):
            sorted_level_1_files.append(lines[i][:-1])
           
   

    # for i in range(len(chr_order)):
    #     for j in range(len(level_1_files)):
    #         file.write(level_1_files[j]+"\n")

    level_1_files = sorted_level_1_files
    # print("Files under summaries : ", len(level_1_files))
    write_file.write("Files under summaries : "+str(len(level_1_files))+"\n")

    # for i in range(5,6):
    for i in range(len(level_1_files)):
        level_2_files = list(f[level_1][level_1_files[i]].keys())
        # print(level_2_files)
        # print("Files under ",level_1_files[i]," : ",len(level_2_files))
        write_file.write(str(i+1)+" : Files under "+level_1_files[i]+" : "+str(len(level_2_files))+"\n")

        write_file.write("------------------------------------------------------------------------------------------\n")
        for j in range(len(level_2_files)):
            level_3_files = f[level_1][level_1_files[i]][level_2_files[j]]
            # print(level_3_files)
            # print("Files under ",level_2_files[j]," : ",len(level_3_files))
           
            leaf_items = list(level_3_files)
          
            if j == 5 :
                write_file.write("\t\t "+str(j+1)+" : Files under "+level_2_files[j]+" : "+str(len(level_3_files))+"\n")
               
                for l in range(len(leaf_items)):
                        
                        sample_data.append(leaf_items[l])

                       
                        xx = np.array(leaf_items[l])
                        write_file.write("--------------------------------------------------------------------------------------------------------------------------------------\n")            
                        write_file.write(f'{str(l+1):<4}'+ " | ")
                        for gg in range(len(xx[0])):
                            write_file.write(f'{str(gg+1):<4}'+ " ")
 
                        write_file.write("\n--------------------------------------------------------------------------------------------------------------------------------------\n")            
                        for out in range(len(xx)):
                            write_file.write(f'{str(out+1):<4}'+ " | ")
                            for inn in range(len(xx[out])):
                                write_file.write(f'{str(xx[out][inn]):<4}'+ " ")
                            write_file.write("\n")
                        # write_file.write("------------------------------------------------------------------------------------------\n")                  
                           
   

print("total number of 2d arrays : ",len(sample_data))
print("class type : ",class_type)

if class_type == 1:
    file_name = "positive.npz"
else:
    file_name = "negative.npz"	

sample_label = [class_type]*len(sample_data)
np.savez_compressed(file_name, data=sample_data, label=sample_label)

print("file name : ",file_name," saved")



    