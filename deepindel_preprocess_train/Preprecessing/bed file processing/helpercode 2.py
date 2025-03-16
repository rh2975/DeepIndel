
file_name = "data.vcf"


file = open(file_name, "r")
write_file = open("changed_vcf.vcf", "w")
lines = file.readlines()

for i in range(len(lines)):
    
    ll = lines[i]

    if ll.startswith("#"):
        # print(ll)
        pass
    else:
        #   print(ll) 
        ll = "chr"+ll

        # print(ll)
    write_file.write(ll)

