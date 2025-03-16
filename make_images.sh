INPUT_DIR="/media/hp/CE62638F62637ADB/sifat_rahat/Dysgu/data/HG001_required_files"
OUTPUT_DIR="${PWD}/Outputs"

TRUTH_BED=${INPUT_DIR}/del.bed
TRUTH_VCF=${INPUT_DIR}/my_deletions.vcf.gz
REF=${INPUT_DIR}/hs37d5.fa

# OUTPUT DIRECTORIES WHERE TRAINING EXAMPLES WILL BE SAVED
TRAIN_OUTPUT=${INPUT_DIR}/PEPPER_TRAIN_IMAGES
TEST_OUTPUT=${INPUT_DIR}/PEPPER_TEST_IMAGES

THREADS=1

for coverage in 50
do
  BAM=${INPUT_DIR}/sorted_testChaged.bam

  echo ${BAM}

  time pepper_variant_train make_train_images \
  -b ${BAM} \
  -f ${REF} \
  -tv ${TRUTH_VCF} \
  -r chr1-2 \
  -rb ${TRUTH_BED} \
  -t ${THREADS} \
  -o ${TEST_OUTPUT} \
  -d 1.0 \
  -p 1.0 \
  --hifi  
done
