library(bedr)

in_str <- "./bed_files/our.bed"
#in_str <- "testBed.bed"

x <- read.table(in_str)

#additional suggested commands

x <- bedr.sort.region(x)

x <- bedr.merge.region(x)



bed2vcf(x, filename = "my_deletions3.vcf", zero.based = TRUE, header = NULL, fasta = "./bed_files/my3.fa")

#bed2vcf(x, filename = "deletions.vcf", zero.based = TRUE, header = NULL, fasta = NULL)
