//
// Created by Kishwar Shafin on 3/21/21.
//

#include "region_summary.h"

#include <utility>

RegionalSummaryGenerator::RegionalSummaryGenerator(string contig, long long region_start, long long region_end, string reference_sequence) {
    this->contig = contig;
    this->ref_start = region_start;
    this->ref_end = region_end;
    this->reference_sequence = std::move(reference_sequence);
    this->total_observered_insert_bases = 0;
    this->max_observed_insert.resize(region_end-region_start+1, 0);
    this->cumulative_observed_insert.resize(region_end-region_start+1, 0);
}

void RegionalSummaryGenerator::generate_max_insert_observed(const type_read& read) {
    int read_index = 0;
    long long ref_position = read.pos;
    int cigar_index = 0;
    long long reference_index;
    if(ImageOptionsRegion::GENERATE_INDELS == true) {
        for (auto &cigar: read.cigar_tuples) {
            if (ref_position > ref_end) break;
            switch (cigar.operation) {
                case CIGAR_OPERATIONS::EQUAL:
                case CIGAR_OPERATIONS::DIFF:
                case CIGAR_OPERATIONS::MATCH:
                    cigar_index = 0;
                    if (ref_position < ref_start) {
                        cigar_index = min(ref_start - ref_position, (long long) cigar.length);
                        read_index += cigar_index;
                        ref_position += cigar_index;
                    }
                    for (int i = cigar_index; i < cigar.length; i++) {
                        reference_index = ref_position - ref_start;
                        read_index += 1;
                        ref_position += 1;
                    }
                    break;
                case CIGAR_OPERATIONS::IN:
                    reference_index = ref_position - ref_start - 1;

                    if (ref_position - 1 >= ref_start &&
                        ref_position - 1 <= ref_end) {
                        max_observed_insert[reference_index] = std::max(max_observed_insert[reference_index], (uint64_t) cigar.length);
                    }
                    read_index += cigar.length;
                    break;
                case CIGAR_OPERATIONS::REF_SKIP:
                case CIGAR_OPERATIONS::PAD:
                case CIGAR_OPERATIONS::DEL:
                    reference_index = ref_position - ref_start - 1;
                    ref_position += cigar.length;
                    break;
                case CIGAR_OPERATIONS::SOFT_CLIP:
                    read_index += cigar.length;
                    break;
                case CIGAR_OPERATIONS::HARD_CLIP:
                    break;
            }
        }
    }
}


void RegionalSummaryGenerator::generate_max_insert_summary(vector <type_read> &reads) {
    for (auto &read:reads) {
        // this populates base_summaries and insert_summaries dictionaries
        generate_max_insert_observed(read);
    }

    cumulative_observed_insert[0] = 0;
    total_observered_insert_bases += max_observed_insert[0];

    positions.push_back(ref_start);
    index.push_back(0);

    for(int j=1; j <= max_observed_insert[0]; j++) {
        positions.push_back(ref_start);
        index.push_back(j);
    }

    for(int i=1;i < max_observed_insert.size(); i++) {
        cumulative_observed_insert[i] = cumulative_observed_insert[i-1] + max_observed_insert[i-1];
        total_observered_insert_bases += max_observed_insert[i];
        positions.push_back(ref_start + i);
        index.push_back(0);
        for(int j=1; j <= max_observed_insert[i]; j++) {
            positions.push_back(ref_start + i);
            index.push_back(j);
        }
    }
}


char check_truth_base(char base) {
    if(base=='A' || base=='a' ||
       base=='C' || base=='c' ||
       base=='T' || base=='t' ||
       base=='G' || base=='g' ||
       base=='*' || base=='#') return base;
    return '*';
}

uint8_t get_label_index(char base_h1, char base_h2) {
    base_h1 = toupper(base_h1);
    base_h2 = toupper(base_h2);
    vector<string> base_labels {"RR", "RA", "RC", "RT", "RG", "R*", "R#", "AA", "AC", "AT", "AG", "A*", "A#", "CC", "CT", "CG", "C*", "C#", "TT", "TG", "T*", "T#", "GG", "G*", "G#", "**", "*#", "##"};


    for(int i=0; i<base_labels.size(); i++) {
        if(base_h1 == base_labels[i][0] && base_h2 == base_labels[i][1]) return i;
        if(base_h2 == base_labels[i][0] && base_h1 == base_labels[i][1]) return i;
    }
    cout<<"TYPE NOT FOUND FOR: "<<base_h1<<" "<<base_h2<<endl;
    return 0;
}


uint8_t get_variant_type_label_index(int type_h1, int type_h2) {

    if (type_h1 == VariantTypes::HOM_REF && type_h2 == VariantTypes::HOM_REF) return 0;

    if (type_h1 == VariantTypes::HOM_REF && type_h2 == VariantTypes::SNP) return 1;
    if (type_h1 == VariantTypes::SNP && type_h2 == VariantTypes::HOM_REF) return 1;

    if (type_h1 == VariantTypes::HOM_REF && type_h2 == VariantTypes::INSERT) return 2;
    if (type_h1 == VariantTypes::INSERT && type_h2 == VariantTypes::HOM_REF) return 2;

    if (type_h1 == VariantTypes::HOM_REF && type_h2 == VariantTypes::DELETE) return 3;
    if (type_h1 == VariantTypes::DELETE && type_h2 == VariantTypes::HOM_REF) return 3;

    if (type_h1 == VariantTypes::SNP && type_h2 == VariantTypes::SNP) return 4;

    if (type_h1 == VariantTypes::SNP && type_h2 == VariantTypes::INSERT) return 5;
    if (type_h1 == VariantTypes::INSERT && type_h2 == VariantTypes::SNP) return 5;

    if (type_h1 == VariantTypes::SNP && type_h2 == VariantTypes::DELETE) return 6;
    if (type_h1 == VariantTypes::DELETE && type_h2 == VariantTypes::SNP) return 6;

    if (type_h1 == VariantTypes::INSERT && type_h2 == VariantTypes::INSERT) return 7;

    if (type_h1 == VariantTypes::INSERT && type_h2 == VariantTypes::DELETE) return 8;
    if (type_h1 == VariantTypes::DELETE && type_h2 == VariantTypes::INSERT) return 8;

    if (type_h1 == VariantTypes::DELETE && type_h2 == VariantTypes::DELETE) return 9;

    cout<<"ERROR: VARIANT LABEL NOT DEFINED: "<<type_h1<<" "<<type_h2<<endl;
    exit(1);
}


int RegionalSummaryGenerator::get_reference_feature_index(char base) {
    base = toupper(base);
    if (base == 'A') return ImageOptionsRegion::REFERENCE_INDEX_START;
    if (base == 'C') return ImageOptionsRegion::REFERENCE_INDEX_START + 1;
    if (base == 'G') return ImageOptionsRegion::REFERENCE_INDEX_START + 2;
    if (base == 'T') return ImageOptionsRegion::REFERENCE_INDEX_START + 3;
    return ImageOptionsRegion::REFERENCE_INDEX_START + 4;
}

int RegionalSummaryGenerator::get_reference_feature_value(char base) {
    base = toupper(base);
    if (base == 'A') return 1;
    if (base == 'C') return 2;
    if (base == 'G') return 3;
    if (base == 'T') return 4;
    return 5;
}

void RegionalSummaryGenerator::encode_reference_bases(vector< vector<int> >& image_matrix) {
    for (long long ref_position = ref_start; ref_position <= ref_end; ref_position++) {
        // encode the C base
        int base_index = (int) (ref_position - ref_start + cumulative_observed_insert[ref_position - ref_start]);
        // int feature_index = get_reference_feature_index(reference_sequence[ref_position - ref_start]);
        int feature_index = 0;
        int value = get_reference_feature_value(reference_sequence[ref_position - ref_start]);
        image_matrix[base_index][feature_index] = value;

        for(int i = 1; i <= max_observed_insert[ref_position - ref_start]; i++) {
            base_index = (int) (ref_position - ref_start + cumulative_observed_insert[ref_position - ref_start]) + i;
            // feature_index = get_reference_feature_index('*');
             cout<<"\t\t\t\t\t\t\t"<<base_index<<endl;
            feature_index = 0;
            value = get_reference_feature_value(reference_sequence[ref_position - ref_start]);
            image_matrix[base_index][feature_index] = value;
        }
    }
}

bool check_ref_base(char base) {
    if(base=='A' || base=='a' ||
       base=='C' || base=='c' ||
       base=='T' || base=='t' ||
       base =='G' || base=='g') return true;
    return false;
}

int RegionalSummaryGenerator::get_feature_index(char ref_base, char base, bool is_reverse) {
    base = toupper(base);
    ref_base = toupper(ref_base);
    bool valid_ref = check_ref_base(ref_base);
    if(valid_ref) {
        // this is a mismatch situation
        if (!is_reverse) {
            int start_index  = 7;
            if (base == 'A') return start_index + 1;
            if (base == 'C') return start_index + 2;
            if (base == 'G') return start_index + 3;
            if (base == 'T') return start_index + 4;
            if (base == 'I') return start_index + 5;
            if (base == 'D') return start_index + 6;
            return start_index + 7;
        } else {
            // tagged and forward
            int start_index  = 18;
            if (base == 'A') return start_index + 1;
            if (base == 'C') return start_index + 2;
            if (base == 'G') return start_index + 3;
            if (base == 'T') return start_index + 4;
            if (base == 'I') return start_index + 5;
            if (base == 'D') return start_index + 6;
            return start_index + 7;
        }
    } else {
        return -1;
    }
}



void RegionalSummaryGenerator::generate_labels(const vector<type_truth_record>& hap1_records, const vector<type_truth_record>& hap2_records) {
    int region_size = (int) (ref_end - ref_start + total_observered_insert_bases + 1);
    labels_hp1.resize(region_size + 1, '*');
    labels_hp2.resize(region_size + 1, '*');
    variant_type_labels_hp1.resize(region_size + 1, VariantTypes::HOM_REF);
    variant_type_labels_hp2.resize(region_size + 1, VariantTypes::HOM_REF);

    hp1_truth_alleles.resize(region_size + 1);
    hp2_truth_alleles.resize(region_size + 1);

    for(long long pos = ref_start; pos <= ref_end; pos++) {
        int base_index = (int)(pos - ref_start + cumulative_observed_insert[pos - ref_start]);
        labels_hp1[base_index] = 'R';
        labels_hp2[base_index] = 'R';
    }

    for(const auto& truth_record : hap1_records) {

        if(truth_record.ref.length() > truth_record.alt.length()) {
            //it's a delete
//            cout<<"TRUTH DELETE: "<<truth_record.contig<<" "<<truth_record.pos_start<<" "<<truth_record.ref<<" "<<truth_record.alt<<endl;
            if (truth_record.pos_start >= ref_start && truth_record.pos_start <= ref_end) {
                int base_index = (int) (truth_record.pos_start - ref_start + cumulative_observed_insert[truth_record.pos_start - ref_start]);
                variant_type_labels_hp1[base_index] = VariantTypes::DELETE;
                labels_hp1[base_index] = '#';
                hp1_truth_alleles[base_index].push_back(truth_record);
            }
        } else if(truth_record.ref.length() < truth_record.alt.length()) {
            //it's an insert
//            cout<<"TRUTH INSERT: "<<truth_record.contig<<" "<<truth_record.pos_start<<" "<<truth_record.ref<<" "<<truth_record.alt<<endl;
            if (truth_record.pos_start >= ref_start && truth_record.pos_start <= ref_end) {
                int base_index = (int) (truth_record.pos_start - ref_start + cumulative_observed_insert[truth_record.pos_start - ref_start]);
                variant_type_labels_hp1[base_index] = VariantTypes::INSERT;
                labels_hp1[base_index] = '*';
                hp1_truth_alleles[base_index].push_back(truth_record);
            }
        } else if(truth_record.ref.length() == truth_record.alt.length()) {
            //it's a SNP
//            cout<<"TRUTH SNP: "<<truth_record.contig<<" "<<truth_record.pos_start<<" "<<truth_record.ref<<" "<<truth_record.alt<<endl;
            if (truth_record.pos_start >= ref_start && truth_record.pos_start <= ref_end) {
                int base_index = (int) (truth_record.pos_start - ref_start + cumulative_observed_insert[truth_record.pos_start - ref_start]);
                variant_type_labels_hp1[base_index] = VariantTypes::SNP;
                hp1_truth_alleles[base_index].push_back(truth_record);
            }

            for(long long pos = truth_record.pos_start; pos < truth_record.pos_end; pos++) {
                if (pos >= ref_start && pos <= ref_end) {
                    int base_index = (int) (pos - ref_start + cumulative_observed_insert[pos - ref_start]);
                    char ref_base = reference_sequence[pos - ref_start];
                    char alt_base = truth_record.alt[pos - truth_record.pos_start];
                    if(ref_base == alt_base) {
                        labels_hp1[base_index] = 'R';
                    } else {
                        labels_hp1[base_index] = truth_record.alt[pos - truth_record.pos_start];
                    }
                }
            }
        }
    }

    for(const auto& truth_record : hap2_records) {
        if(truth_record.ref.length() > truth_record.alt.length()) {
            //it's a delete
            if (truth_record.pos_start >= ref_start && truth_record.pos_start <= ref_end) {
                int base_index = (int) (truth_record.pos_start - ref_start + cumulative_observed_insert[truth_record.pos_start - ref_start]);
                variant_type_labels_hp2[base_index] = VariantTypes::DELETE;
                labels_hp2[base_index] = '#';
                hp2_truth_alleles[base_index].push_back(truth_record);
            }
        } else if(truth_record.ref.length() < truth_record.alt.length()) {
            //it's an insert
            if (truth_record.pos_start >= ref_start && truth_record.pos_start <= ref_end) {
                int base_index = (int) (truth_record.pos_start - ref_start + cumulative_observed_insert[truth_record.pos_start - ref_start]);
                variant_type_labels_hp2[base_index] = VariantTypes::INSERT;
                labels_hp2[base_index] = '*';
                hp2_truth_alleles[base_index].push_back(truth_record);
            }
        } else if(truth_record.ref.length() == truth_record.alt.length()) {
            //it's a SNP
            if (truth_record.pos_start >= ref_start && truth_record.pos_start <= ref_end) {
                int base_index = (int) (truth_record.pos_start - ref_start + cumulative_observed_insert[truth_record.pos_start - ref_start]);
                variant_type_labels_hp2[base_index] = VariantTypes::SNP;
                hp2_truth_alleles[base_index].push_back(truth_record);
            }

            for(long long pos = truth_record.pos_start; pos < truth_record.pos_end; pos++) {
                if (pos >= ref_start && pos <= ref_end) {
                    int base_index = (int) (pos - ref_start + cumulative_observed_insert[pos - ref_start]);

                    char ref_base = reference_sequence[pos - ref_start];
                    char alt_base = truth_record.alt[pos - truth_record.pos_start];
                    if(ref_base == alt_base) {
                        labels_hp2[base_index] = 'R';
                    } else {
                        labels_hp2[base_index] = truth_record.alt[pos - truth_record.pos_start];
                    }
                }
            }
        }
    }
}


void RegionalSummaryGenerator::populate_summary_matrix(vector< vector<int> >& image_matrix,
                                                       int *coverage_vector,
                                                       int *snp_count,
                                                       int *insert_count,
                                                       int *delete_count,
                                                       vector< map<string, int> > &AlleleFrequencyMap,
                                                       vector< map<string, int> > &AlleleFrequencyMapFwdStrand,
                                                       vector< map<string, int> > &AlleleFrequencyMapRevStrand,
                                                       vector< set<string> > &AlleleMap,
                                                       type_read read,
                                                       double min_snp_baseq,
                                                       double min_indel_baseq) {
    int read_index = 0;
    long long ref_position = read.pos;
    int cigar_index = 0;
    double base_quality = read.base_qualities[read_index];
    for (int cigar_i=0; cigar_i<read.cigar_tuples.size(); cigar_i++) {
        CigarOp cigar = read.cigar_tuples[cigar_i];
        if (ref_position > ref_end) break;
        switch (cigar.operation) {
            case CIGAR_OPERATIONS::EQUAL:
            case CIGAR_OPERATIONS::DIFF:
            case CIGAR_OPERATIONS::MATCH:
                cigar_index = 0;
                if (ref_position < ref_start) {
                    cigar_index = min(ref_start - ref_position, (long long) cigar.length);
                    read_index += cigar_index;
                    ref_position += cigar_index;
                }
                for (int i = cigar_index; i < cigar.length; i++) {
                    base_quality = read.base_qualities[read_index];

                    if (ref_position >= ref_start && ref_position <= ref_end) {
                        char base = read.sequence[read_index];
                        char ref_base = reference_sequence[ref_position - ref_start];
                        string alt(1, read.sequence[read_index]);

                        int base_index = (int)(ref_position - ref_start + cumulative_observed_insert[ref_position - ref_start]);
                        int feature_index = get_feature_index(ref_base, base, read.flags.is_reverse);

                        // update the summary of base
                        if(base_quality >= min_snp_baseq) {
                            coverage_vector[ref_position - ref_start] += 1;
                            // look front and see if it's anchoring an INSERT or DELETE
                            if(i == cigar.length - 1 && cigar_i != read.cigar_tuples.size() - 1) {
                                CigarOp next_cigar = read.cigar_tuples[cigar_i + 1];
                                if(next_cigar.operation != CIGAR_OPERATIONS::IN && next_cigar.operation != CIGAR_OPERATIONS::DEL) {
                                    if(!read.flags.is_reverse) image_matrix[base_index][4] -= 1;
                                    else image_matrix[base_index][15] -= 1;
                                }
                            }
                            else {
                                if (!read.flags.is_reverse) image_matrix[base_index][4] -= 1;
                                else image_matrix[base_index][15] -= 1;
                            }
                        }

                        if(ref_base != base && base_quality >= min_snp_baseq) {
                            snp_count[ref_position - ref_start] += 1;
                            if(feature_index >= 0) image_matrix[base_index][feature_index] -= 1;
                            // save the candidate
                            string candidate_string = char(AlleleType::SNP_ALLELE + '0') + alt;

                            int region_index = (int) (ref_position - ref_start);

                            if (AlleleFrequencyMap[region_index].find(candidate_string) != AlleleFrequencyMap[region_index].end()) {
                                AlleleFrequencyMap[region_index][candidate_string] += 1;
                                if(read.flags.is_reverse) {
                                    AlleleFrequencyMapRevStrand[region_index][candidate_string] += 1;
                                } else {
                                    AlleleFrequencyMapFwdStrand[region_index][candidate_string] += 1;
                                }
                            } else {
                                AlleleFrequencyMap[region_index][candidate_string] = 1;
                                if(read.flags.is_reverse) {
                                    AlleleFrequencyMapFwdStrand[region_index][candidate_string] = 0;
                                    AlleleFrequencyMapRevStrand[region_index][candidate_string] = 1;
                                } else {
                                    AlleleFrequencyMapFwdStrand[region_index][candidate_string] = 1;
                                    AlleleFrequencyMapRevStrand[region_index][candidate_string] = 0;
                                }
                            }

                            if (AlleleMap[region_index].find(candidate_string) == AlleleMap[region_index].end())
                                AlleleMap[region_index].insert(candidate_string);
                        } else if (base_quality >= min_snp_baseq){
                            if(feature_index >= 0)  image_matrix[base_index][feature_index] -= 1;

                        }
                    }
                    read_index += 1;
                    ref_position += 1;
                }
                break;
            case CIGAR_OPERATIONS::IN:
                if (ref_position - 1 >= ref_start && ref_position - 1 <= ref_end && read_index - 1 >= 0) {
                    // process insert allele here
                    string alt;
                    char ref_base = reference_sequence[ref_position - 1 - ref_start];
                    int base_index = (int)((ref_position - 1) - ref_start + cumulative_observed_insert[(ref_position - 1) - ref_start]);
                    int insert_count_index =  get_feature_index(ref_base, 'I', read.flags.is_reverse);

                    if (read_index - 1 >= 0) alt = read.sequence.substr(read_index - 1, cigar.length + 1);
                    else alt = ref_base + read.sequence.substr(read_index, cigar.length);

                    int len = cigar.length + 1;
                    base_quality = 0;
                    int start_index = 0;
                    if (read_index - 1 >= 0) start_index  = read_index - 1;
                    else start_index = read_index;

                    for(int i = start_index; i < start_index + len; i++) {
                        base_quality += read.base_qualities[i];
                    }

                    // include reads that were excluded due to anchor bases' quality
                    if(base_quality >= min_indel_baseq * len && read.base_qualities[start_index] < min_snp_baseq)
                        coverage_vector[ref_position - 1 - ref_start] += 1;


                    // save the candidate
                    string candidate_string = char(AlleleType::INSERT_ALLELE + '0') + alt;

                    // only process candidates that are smaller than 50bp as they 50bp+ means SV
                    if(candidate_string.length() <= 61 && base_quality >= min_indel_baseq * len) {
                        if(insert_count_index >= 0) image_matrix[base_index][insert_count_index] -= 1;

                        insert_count[ref_position - 1 - ref_start] += 1;
                        int region_index = (int) (ref_position - 1 - ref_start);

                        if (AlleleFrequencyMap[region_index].find(candidate_string) != AlleleFrequencyMap[region_index].end()) {
                            AlleleFrequencyMap[region_index][candidate_string] += 1;
                            if(read.flags.is_reverse) {
                                AlleleFrequencyMapRevStrand[region_index][candidate_string] += 1;
                            } else {
                                AlleleFrequencyMapFwdStrand[region_index][candidate_string] += 1;
                            }
                        } else {
                            AlleleFrequencyMap[region_index][candidate_string] = 1;
                            if(read.flags.is_reverse) {
                                AlleleFrequencyMapFwdStrand[region_index][candidate_string] = 0;
                                AlleleFrequencyMapRevStrand[region_index][candidate_string] = 1;
                            } else {
                                AlleleFrequencyMapFwdStrand[region_index][candidate_string] = 1;
                                AlleleFrequencyMapRevStrand[region_index][candidate_string] = 0;
                            }
                        }

                        if (AlleleMap[region_index].find(candidate_string) == AlleleMap[region_index].end())
                            AlleleMap[region_index].insert(candidate_string);
                    }
                }
                read_index += cigar.length;
                break;
            case CIGAR_OPERATIONS::DEL:
                // process delete allele here
                if (ref_position -1 >= ref_start && ref_position - 1 <= ref_end) {
                    char ref_base = reference_sequence[ref_position - 1 - ref_start];
                    int base_index = (int)(ref_position - 1 - ref_start + cumulative_observed_insert[ref_position - 1 - ref_start]);
                    int delete_count_index =  get_feature_index(ref_base, 'D', read.flags.is_reverse);
                    if(delete_count_index >= 0) image_matrix[base_index][delete_count_index] -= 1.0;

                    // process delete allele here
                    string ref = reference_sequence.substr(ref_position - ref_start - 1, cigar.length + 1);
                    string alt;

                    if (read_index - 1 >= 0) alt = read.sequence.substr(read_index - 1, 1);
                    else alt = reference_sequence.substr(ref_position - ref_start - 1, 1);

                    base_quality = read.base_qualities[read_index];
                    string candidate_string = char(AlleleType::DELETE_ALLELE + '0') + ref;

                    // only process candidates that are smaller than 50bp as they 50bp+ means SV
                    // no base-quality check for deletes
                    if(candidate_string.length() <= 61) {
                        delete_count[ref_position - 1 - ref_start] += 1;
                        int region_index = (int) (ref_position - 1 - ref_start);

                        if (AlleleFrequencyMap[region_index].find(candidate_string) != AlleleFrequencyMap[region_index].end()) {
                            AlleleFrequencyMap[region_index][candidate_string] += 1;
                            if(read.flags.is_reverse) {
                                AlleleFrequencyMapRevStrand[region_index][candidate_string] += 1;
                            } else {
                                AlleleFrequencyMapFwdStrand[region_index][candidate_string] += 1;
                            }
                        } else {
                            AlleleFrequencyMap[region_index][candidate_string] = 1;
                            if(read.flags.is_reverse) {
                                AlleleFrequencyMapFwdStrand[region_index][candidate_string] = 0;
                                AlleleFrequencyMapRevStrand[region_index][candidate_string] = 1;
                            } else {
                                AlleleFrequencyMapFwdStrand[region_index][candidate_string] = 1;
                                AlleleFrequencyMapRevStrand[region_index][candidate_string] = 0;
                            }
                        }

                        if (AlleleMap[region_index].find(candidate_string) == AlleleMap[region_index].end())
                            AlleleMap[region_index].insert(candidate_string);
                    }

                    // cout<<"DEL: "<<ref_position<<" "<<ref<<" "<<alt<<" "<<AlleleFrequencyMap[candidate_alt]<<endl;

                }
                // dont' expand to the full delete length, rather just mount everything to the anchor

                for (int i = 0; i < cigar.length; i++) {
                    if (ref_position + i >= ref_start && ref_position + i <= ref_end) {
                        // update the summary of base
                        int base_index = (int) (ref_position - ref_start + i + cumulative_observed_insert[ref_position - ref_start + i]);
                        char ref_base = reference_sequence[ref_position - ref_start + i];
                        int feature_index = get_feature_index(ref_base, '*', read.flags.is_reverse);

                        if(feature_index >= 0)  image_matrix[base_index][feature_index] -= 1;
//                        coverage_vector[ref_position - ref_start + i] += 1.0;
                    }
                }

                ref_position += cigar.length;
                break;
            case CIGAR_OPERATIONS::REF_SKIP:
            case CIGAR_OPERATIONS::PAD:
                ref_position += cigar.length;
            case CIGAR_OPERATIONS::SOFT_CLIP:
                read_index += cigar.length;
                break;
            case CIGAR_OPERATIONS::HARD_CLIP:
                break;
        }
    }
}

vector<CandidateImageSummary> RegionalSummaryGenerator::generate_summary(vector <type_read> &reads,
                                                                         double min_snp_baseq,
                                                                         double min_indel_baseq,
                                                                         double snp_freq_threshold,
                                                                         double insert_freq_threshold,
                                                                         double delete_freq_threshold,
                                                                         double min_coverage_threshold,
                                                                         double snp_candidate_freq_threshold,
                                                                         double indel_candidate_freq_threshold,
                                                                         double candidate_support_threshold,
                                                                         bool skip_indels,
                                                                         long long candidate_region_start,
                                                                         long long candidate_region_end,
                                                                         int candidate_window_size,
                                                                         int feature_size,
                                                                         bool train_mode) {
    int region_size = (int) (ref_end - ref_start + total_observered_insert_bases + 1);
    // Generate a cover vector of chunk size. Chunk size = 10kb defining the region
    int coverage_vector[ref_end - ref_start + 1];
    int snp_count[ref_end - ref_start + 1];
    int insert_count[ref_end - ref_start + 1];
    int delete_count[ref_end - ref_start + 1];
    vector< map<string, int> > AlleleFrequencyMap;
    vector< map<string, int> > AlleleFrequencyMapFwdStrand;
    vector< map<string, int> > AlleleFrequencyMapRevStrand;
    vector< set<string> > AlleleMap;

    // generate the image matrix of chunk_size (10kb) * feature_size (10)
    vector< vector<int> > image_matrix;

    image_matrix.resize(region_size + 1, vector<int>(feature_size));
    AlleleFrequencyMap.resize(region_size + 1);
    AlleleFrequencyMapFwdStrand.resize(region_size + 1);
    AlleleFrequencyMapRevStrand.resize(region_size + 1);
    AlleleMap.resize(region_size + 1);

    for (int i = 0; i < region_size + 1; i++) {
        for (int j = 0; j < feature_size; j++)
            image_matrix[i][j] = 0;
    }

    memset(coverage_vector, 0, sizeof(coverage_vector));
    memset(snp_count, 0, sizeof(snp_count));
    memset(insert_count, 0, sizeof(insert_count));
    memset(delete_count, 0, sizeof(delete_count));

    encode_reference_bases(image_matrix);

    // now iterate over all of the reads and populate the image matrix and coverage matrix
    for (auto &read:reads) {
        // this populates base_summaries and insert_summaries dictionaries
        if(read.mapping_quality > 0) {
            populate_summary_matrix(image_matrix, coverage_vector, snp_count, insert_count, delete_count,
                                    AlleleFrequencyMap, AlleleFrequencyMapFwdStrand, AlleleFrequencyMapRevStrand, AlleleMap, read, min_snp_baseq, min_indel_baseq);
        }
    }
  
    vector<long long> filtered_candidate_positions;
    bool snp_threshold_pass[ref_end - ref_start + 1];
    bool insert_threshold_pass[ref_end - ref_start + 1];
    bool delete_threshold_pass[ref_end - ref_start + 1];
    memset(snp_threshold_pass, 0, sizeof(snp_threshold_pass));
    memset(insert_threshold_pass, 0, sizeof(insert_threshold_pass));
    memset(delete_threshold_pass, 0, sizeof(delete_threshold_pass));

    // once the image matrix is generated, scale the counted values.
    for(int i=0;i<region_size;i++){
        double snp_fraction = snp_count[positions[i]-ref_start] / max(1.0, (double) coverage_vector[positions[i]-ref_start]);
        double insert_fraction = insert_count[positions[i]-ref_start] / max(1.0, (double) coverage_vector[positions[i]-ref_start]);
        double delete_fraction = delete_count[positions[i]-ref_start] / max(1.0, (double) coverage_vector[positions[i]-ref_start]);
        // cout<<"snp_frac: "<<snp_fraction<<" insert_frac "<<insert_fraction<<" delete_frac: "<<delete_fraction<<endl;
        // cout<<"snp_freq_t: "<<snp_freq_threshold<<" insert_freq_t: "<<insert_freq_threshold<<" delete_freq_t: "<<delete_freq_threshold<<endl;	
        if(snp_fraction >= snp_freq_threshold || insert_fraction >= insert_freq_threshold || delete_fraction >= delete_freq_threshold) {
            // cout<<"\t\t\t\t\t aschilam ekahne 1...\n";
        
            // if(positions[i] >= candidate_region_start && positions[i] <= candidate_region_end && coverage_vector[positions[i]-ref_start] >= min_coverage_threshold) {
                filtered_candidate_positions.push_back(positions[i]);
                if(snp_fraction >= snp_freq_threshold) snp_threshold_pass[positions[i] - ref_start] = true;
                if(insert_fraction >= insert_freq_threshold) insert_threshold_pass[positions[i] - ref_start] = true;
                if(delete_fraction >= delete_freq_threshold) delete_threshold_pass[positions[i] - ref_start] = true;
            // }
        }

        for(int j=ImageOptionsRegion::BASE_INDEX_START; j < ImageOptionsRegion::BASE_INDEX_START + ImageOptionsRegion::BASE_INDEX_SIZE ; j++){
            if(image_matrix[i][j] >= 0)
                image_matrix[i][j] = (int) min(image_matrix[i][j], ImageOptionsRegion::MAX_COLOR_VALUE);
            else
                image_matrix[i][j] = (int) max(image_matrix[i][j], ImageOptionsRegion::MIN_COLOR_VALUE);
        }
    }


    labels.resize(region_size + 1, 0);
    labels_variant_type.resize(region_size + 1, 0);
    // check if train mode, if yes, then generate labels
    if(train_mode) {
        for (int i = 0; i < labels_hp1.size(); i++) {
            labels[i] = get_label_index(labels_hp1[i], labels_hp2[i]);
            labels_variant_type[i] = get_variant_type_label_index(variant_type_labels_hp1[i], variant_type_labels_hp2[i]);
        }
    }

    vector<CandidateImageSummary> all_candidate_images;
    // at this point all of the images are generated. So we can create the images for each candidate position.
    for(long long candidate_position : filtered_candidate_positions) {
        // cout<<"\tCANDIDATE POSITION : "<<candidate_position<<endl;
        for (auto it=AlleleMap[candidate_position - ref_start].begin(); it!=AlleleMap[candidate_position - ref_start].end(); ++it) {
            CandidateImageSummary candidate_summary;
            candidate_summary.contig = contig;
            candidate_summary.position = candidate_position;
            bool debug = 0;
            if(debug) {
                // cout << "-------------------------START----------------------------------------" << endl;
                cout << "Candidate position 1: " << candidate_position << endl;
                // cout << "Coverage: " << coverage_vector[candidate_position - ref_start] << endl;
                // cout << "Candidates: " << endl;
                debug = 0;
            }

            candidate_summary.depth = min(coverage_vector[candidate_position-ref_start], ImageOptionsRegion::MAX_COLOR_VALUE);

            string candidate_string = *it;

            int allele_depth = AlleleFrequencyMap[candidate_position - ref_start][candidate_string];
            int allele_depth_fwd = AlleleFrequencyMapFwdStrand[candidate_position - ref_start][candidate_string];
            int allele_depth_rev = AlleleFrequencyMapRevStrand[candidate_position - ref_start][candidate_string];
            double candidate_frequency = ((double) allele_depth / max(1.0, (double) candidate_summary.depth));
//            cout<<candidate_string<<" "<<allele_depth<<" "<<candidate_frequency<<endl;
            string candidate_allele = candidate_string.substr(1, candidate_string.length());
            // minimum 2 reads supporting the candidate or frequency is lower than 10
            
            if (allele_depth < candidate_support_threshold) {
                continue;
            }
            // see if candidate passes the candidate frequency threshold
            if (candidate_string[0] != '1' && candidate_frequency < indel_candidate_freq_threshold) {
                continue;
            }
            if (candidate_string[0] == '1' && candidate_frequency < snp_candidate_freq_threshold) {
                continue;
            }
            // if Candidate is INDEL but we are skipping INDELs
            if ( candidate_string[0] != '1' and skip_indels == true) {
                continue;
            }
            // only pick type-specific candidates for each site
            if((candidate_string[0] == '1' && !snp_threshold_pass[candidate_position - ref_start]) ||
               (candidate_string[0] == '2' && !insert_threshold_pass[candidate_position - ref_start]) ||
               (candidate_string[0] == '3' && !delete_threshold_pass[candidate_position - ref_start])) {
                continue;
            }
            
            int base_index = (int) (candidate_position - ref_start + cumulative_observed_insert[candidate_position - ref_start]);
            int sifat_debug = 0;
            if(sifat_debug) {
                if (candidate_string[0]=='3')
                {
                // cout<<"ref start: "<<ref_start<<" ref end: "<<ref_end<<endl;
                // cout<<"CANDIDATE ALLELE SELECTED: "<<candidate_allele<<endl;
                cout<<"CANDIDATE TYPE: "<<candidate_string[0]<<endl;
                // cout<<"REF BASE: "<<reference_sequence[candidate_position - ref_start]<<endl;
                cout<<"candidate_position : "<<candidate_position<<endl;
                }
                else {

                // cout<<"\t\rref start: "<<ref_start<<" ref end: "<<ref_end<<endl;
                // cout<<"CANDIDATE ALLELE SELECTED: "<<candidate_allele<<endl;
                cout<<"\t\tCANDIDATE TYPE: "<<candidate_string[0]<<endl;
            
                cout<<"\t\tcandidate_position : "<<candidate_position<<endl;

                }
                
            }
            char ref_base = reference_sequence[candidate_position - ref_start];
            if (train_mode) {
                vector<type_truth_record> hp1_truths = hp1_truth_alleles[base_index];
                vector<type_truth_record> hp2_truths = hp2_truth_alleles[base_index];
                vector<string> hp1_alleles;
                vector<string> hp2_alleles;
                for(const auto& truth_record : hp1_truths) {
                    if(truth_record.ref.length() > truth_record.alt.length()) {
                        //it's a delete
                        string alt_allele = truth_record.ref;
                        string ref_allele = truth_record.alt;
                        if(alt_allele.length() > 1 && ref_allele.length() > 1) {
                            int min_length = min(alt_allele.length(), ref_allele.length());
                            alt_allele = alt_allele.substr(0, alt_allele.length() - min_length + 1);
                        }
                        hp1_alleles.push_back(char(AlleleType::DELETE_ALLELE + '0') + alt_allele);
                    } else if(truth_record.ref.length() < truth_record.alt.length()) {
                        //it's an insert
                        string alt_allele = truth_record.alt;
                        string ref_allele = truth_record.ref;
                        if(alt_allele.length() > 1 && ref_allele.length() > 1) {
                            int min_length = min(alt_allele.length(), ref_allele.length());
                            alt_allele = alt_allele.substr(0, alt_allele.length() - min_length + 1);
                        }
                        hp1_alleles.push_back(char(AlleleType::INSERT_ALLELE + '0') + alt_allele);
                    } else if(truth_record.ref.length() == truth_record.alt.length()) {
                        //it's a SNP
                        string alt_allele = truth_record.alt;
                        string ref_allele = truth_record.ref;
                        if(alt_allele.length() > 1 && ref_allele.length() > 1) {
                            int min_length = min(alt_allele.length(), ref_allele.length());
                            alt_allele = alt_allele.substr(0, alt_allele.length() - min_length + 1);
                        }
                        hp1_alleles.push_back(char(AlleleType::SNP_ALLELE + '0') + alt_allele);
                    }
                }
                for(const auto& truth_record : hp2_truths) {
                    if(truth_record.ref.length() > truth_record.alt.length()) {
                        //it's a delete
                        string alt_allele = truth_record.ref;
                        string ref_allele = truth_record.alt;
                        if(alt_allele.length() > 1 && ref_allele.length() > 1) {
//                            cout<<"BEFORE: "<<alt_allele<<endl;
                            int min_length = min(alt_allele.length(), ref_allele.length());
                            alt_allele = alt_allele.substr(0, alt_allele.length() - min_length + 1);
//                            cout<<"AFTER: "<<alt_allele<<endl;
                        }

                        hp2_alleles.push_back(char(AlleleType::DELETE_ALLELE + '0') + alt_allele);
                    } else if(truth_record.ref.length() < truth_record.alt.length()) {
                        //it's an insert
                        string alt_allele = truth_record.alt;
                        string ref_allele = truth_record.ref;
                        if(alt_allele.length() > 1 && ref_allele.length() > 1) {
                            int min_length = min(alt_allele.length(), ref_allele.length());
                            alt_allele = alt_allele.substr(0, alt_allele.length() - min_length + 1);
                        }
                        hp2_alleles.push_back(char(AlleleType::INSERT_ALLELE + '0') + alt_allele);
                    } else if(truth_record.ref.length() == truth_record.alt.length()) {
                        //it's a SNP
                        string alt_allele = truth_record.alt;
                        string ref_allele = truth_record.ref;
                        if(alt_allele.length() > 1 && ref_allele.length() > 1) {
                            int min_length = min(alt_allele.length(), ref_allele.length());
                            alt_allele = alt_allele.substr(0, alt_allele.length() - min_length + 1);
                        }
                        hp2_alleles.push_back(char(AlleleType::SNP_ALLELE + '0') + alt_allele);
                    }
                }
                bool found_in_hp1 = false;
                bool found_in_hp2 = false;
//                cout<<"##############"<<endl;
//                cout<<"Candidate: "<<candidate_allele<<endl;
//                cout<<"HP1 truths:"<<endl;
                for(const auto& alt_hp1 : hp1_alleles) {
//                    cout<<alt_hp1<<endl;
                    if(alt_hp1 == candidate_string) {
                        found_in_hp1 = true;
                    }
                }

//                cout<<"HP2 truths:"<<endl;
                for(const auto& alt_hp2 : hp2_alleles) {
//                    cout<<alt_hp2<<endl;
                    if(alt_hp2 == candidate_string) {
                        found_in_hp2 = true;
                    }
                }
//                cout<<"##############"<<endl;

                int gt_label = 0;
                if(found_in_hp1 && found_in_hp2) {
                    gt_label = 2;
                } else if(found_in_hp1 || found_in_hp2){
                    gt_label = 1;
                }

                candidate_summary.base_label = labels[base_index];
                candidate_summary.type_label = gt_label;
                
                if(debug) {
                    cout << "BASE LABEL: " <<int(candidate_summary.base_label)<<endl;
                    cout << "TYPE LABEL: " <<int(candidate_summary.type_label)<<endl;
                    
                }
            } else {
                candidate_summary.base_label = 0;
                candidate_summary.type_label = 0;
            }
            
            int base_left = base_index - candidate_window_size / 2;
            int base_right = base_index + candidate_window_size / 2;
            // if(sifat_debug)
            //     cout<<"base left: "<<base_left<<" base right: "<<base_right<<" base_index: "<<base_index<<endl;
            // cout<<"--------------------------------------------------\n";
            candidate_summary.image_matrix.resize(candidate_window_size + 1, vector<int>(feature_size));
            // cout<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\t"<<endl;

            // cout << "FEATURE SIZE = " << image_matrix.size() << "----------------------------------------------------------\n\n";

            // now copy the entire feature matrix
            for (int i = base_left; i <= base_right; i++) {
                for (int j = 0; j < feature_size; j++) {
                    if (i < 0 || i > region_size) {
                        candidate_summary.image_matrix[i - base_left][j] = 0;
                    } else {
                        candidate_summary.image_matrix[i - base_left][j] = image_matrix[i][j];
                    }
                }
            }

            // set type-specific feature values
            if(debug) {
                cout << candidate_string << " " << allele_depth << " " << candidate_summary.depth << " " << candidate_frequency << endl;
                cout << snp_threshold_pass[candidate_position - ref_start] << " " << insert_threshold_pass[candidate_position - ref_start] << " " << delete_threshold_pass[candidate_position - ref_start] << endl;
            }
            if (snp_threshold_pass[candidate_position - ref_start] && candidate_string[0] == '1') {
                if(debug) {
                    cout << candidate_string << ",";
                    cout << AlleleFrequencyMap[candidate_position - ref_start][candidate_string] << endl;
                }
                int mid_index = candidate_window_size / 2;
                int forward_feature_index = get_feature_index(ref_base, candidate_string[1], false);
                int reverse_feature_index = get_feature_index(ref_base, candidate_string[1], true);
                candidate_summary.image_matrix[mid_index][1] = get_reference_feature_value(candidate_string[1]); // min((int) candidate_string.length() - 1, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][5] = min(allele_depth_fwd, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][16] = min(allele_depth_rev, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][forward_feature_index] = (-1) * candidate_summary.image_matrix[mid_index][forward_feature_index];
                candidate_summary.image_matrix[mid_index][reverse_feature_index] = (-1) * candidate_summary.image_matrix[mid_index][reverse_feature_index];
                candidate_summary.candidates.push_back(candidate_string);
                candidate_summary.candidate_frequency.push_back(min(allele_depth, ImageOptionsRegion::MAX_COLOR_VALUE));
            } else if (insert_threshold_pass[candidate_position - ref_start] && candidate_string[0] == '2') {
                if(debug) {
                    cout << "INSERT" << " " << candidate_string << ",";
                    cout << AlleleFrequencyMap[candidate_position - ref_start][candidate_string] << endl;
                }
                int mid_index = candidate_window_size / 2;
                int forward_feature_index = get_feature_index(ref_base, 'I', false);
                int reverse_feature_index = get_feature_index(ref_base, 'I', true);
                candidate_summary.image_matrix[mid_index][2] = min((int) candidate_string.length() - 1, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][6] = min(allele_depth_fwd, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][17] = min(allele_depth_rev, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][forward_feature_index] = (-1) * candidate_summary.image_matrix[mid_index][forward_feature_index];
                candidate_summary.image_matrix[mid_index][reverse_feature_index] = (-1) * candidate_summary.image_matrix[mid_index][reverse_feature_index];
                candidate_summary.candidates.push_back(candidate_string);
                candidate_summary.candidate_frequency.push_back(min(allele_depth, ImageOptionsRegion::MAX_COLOR_VALUE));
            } else if (delete_threshold_pass[candidate_position - ref_start] && candidate_string[0] == '3') {
                if(debug) {
                    cout << "DELETE: " << candidate_string << ",";
                    cout << AlleleFrequencyMap[candidate_position - ref_start][candidate_string] << endl;
                }
                int mid_index = candidate_window_size / 2;
                int del_len = (int) candidate_string.length() - 1;
                int end_index = min(mid_index + del_len - 1, candidate_window_size - 1);
                int forward_feature_index = get_feature_index(ref_base, 'D', false);
                int reverse_feature_index = get_feature_index(ref_base, 'D', true);
                candidate_summary.image_matrix[mid_index][3] = min((int) del_len, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][7] = min(allele_depth_fwd, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][18] = min(allele_depth_rev, ImageOptionsRegion::MAX_COLOR_VALUE);
                candidate_summary.image_matrix[mid_index][forward_feature_index] = (-1) * candidate_summary.image_matrix[mid_index][forward_feature_index];
                candidate_summary.image_matrix[mid_index][reverse_feature_index] = (-1) * candidate_summary.image_matrix[mid_index][reverse_feature_index];
                candidate_summary.candidates.push_back(candidate_string);
                candidate_summary.candidate_frequency.push_back(min(allele_depth, ImageOptionsRegion::MAX_COLOR_VALUE));
                // Update the image matrix for the rest of the window
                forward_feature_index = get_feature_index(ref_base, '*', false);
                reverse_feature_index = get_feature_index(ref_base, '*', true);
                for(int idx = mid_index + 1; idx <= end_index; idx++) {
                    candidate_summary.image_matrix[idx][3] = min((int) candidate_string.length() - 1, ImageOptionsRegion::MAX_COLOR_VALUE);
                    candidate_summary.image_matrix[idx][7] = min(allele_depth_fwd, ImageOptionsRegion::MAX_COLOR_VALUE);
                    candidate_summary.image_matrix[idx][18] = min(allele_depth_rev, ImageOptionsRegion::MAX_COLOR_VALUE);
                    candidate_summary.image_matrix[idx][forward_feature_index] = (-1) * candidate_summary.image_matrix[idx][forward_feature_index];
                    candidate_summary.image_matrix[idx][reverse_feature_index] = (-1) * candidate_summary.image_matrix[idx][reverse_feature_index];
                }
            }
            all_candidate_images.push_back(candidate_summary);
            if(debug) {
                debug_candidate_summary(candidate_summary, candidate_window_size, train_mode);
                cout << "-------------------------END----------------------------------------" << endl;
            }
        }
    }


    return all_candidate_images;
}


void RegionalSummaryGenerator::debug_print_matrix(vector<vector<int> > image_matrix, bool train_mode) {
    cout << "------------- IMAGE MATRIX" << endl;

    cout << setprecision(3);
    for (long long i = ref_start; i <= ref_end; i++) {
        if(i==ref_start) cout<<"REF:\t";
        cout << "  " << reference_sequence[i - ref_start] << "\t";
        if (max_observed_insert[i - ref_start] > 0) {
            for (uint64_t ii = 0; ii < max_observed_insert[i - ref_start]; ii++)cout << "  *" << "\t";
        }
    }
    cout << endl;

    if(train_mode) {
        for (int i = 0; i <= ref_end - ref_start + total_observered_insert_bases; i++) {
            if (i == 0) cout << "TRH1:\t";
            cout << "  " << labels_hp1[i] << "\t";
        }
        cout << endl;

        for (int i = 0; i <= ref_end - ref_start + total_observered_insert_bases; i++) {
            if (i == 0) cout << "TRH2:\t";
            cout << "  " << labels_hp2[i] << "\t";
        }
        cout << endl;
    }

    for (int i = 0; i < labels.size(); i++) {
        if(i==0) cout<<"LBL:\t";
        printf("%3d\t", labels[i]);
    }
    cout << endl;

    for (int i = 0; i < labels_variant_type.size(); i++) {
        if(i==0) cout<<"TYP:\t";
        printf("%3d\t", labels_variant_type[i]);
    }
    cout << endl;

    cout<<"POS:\t";
    // for(int i=0; i < positions.size(); i++ ) {
    //     printf("%3lld\t", positions[i] % 100);
    // }
    cout << endl;
    int image_size = ImageOptionsRegion::REFERENCE_INDEX_SIZE + ImageOptionsRegion::SUPPORT_INDEX_SIZE + ImageOptionsRegion::BASE_INDEX_SIZE;
    for (int i = 0; i < image_size; i++) {
        cout<< ImageOptionsRegion::column_values[i] <<"\t";
        int region_size = (int) (ref_end - ref_start + total_observered_insert_bases + 1);

        for (int j = 0; j < region_size; j++) {
            printf("%3d\t", image_matrix[j][i]);
        }
        cout << endl;
    }

}


void RegionalSummaryGenerator::debug_candidate_summary(CandidateImageSummary candidate, int small_chunk_size, bool train_mode) {
    vector<string> decoded_base_lables {"RR", "RA", "RC", "RT", "RG", "R*", "R#", "AA", "AC", "AT", "AG", "A*", "A#", "CC", "CT", "CG", "C*", "C#", "TT", "TG", "T*", "T#", "GG", "G*", "G#", "**", "*#", "##"};
    vector<string> decoded_type_lables {"RR", "RS", "RI", "RD", "SS", "SI", "SD", "II", "ID", "DD" };
    cout << "------------- CANDIDATE PILEUP" << endl;
    cout<<"Contig: "<<candidate.contig<<endl;
    cout<<"Position: "<<candidate.position<<endl;
    cout<<"Type label: "<<(int)candidate.type_label<<" "<<decoded_type_lables[candidate.type_label]<<endl;
    cout<<"Base label: :"<<(int)candidate.base_label<<" "<<decoded_base_lables[candidate.base_label]<<endl;

    long long candidate_ref_start = candidate.position - small_chunk_size / 2;
    long long candidate_ref_end = candidate.position + small_chunk_size / 2;
    cout << setprecision(3);
    for (long long i = candidate_ref_start; i <= candidate_ref_end; i++) {
        if (i == candidate_ref_start) cout << "POS:\t";
        printf("%3lld\t", (i - candidate_ref_start) % 100);
    }
    cout << endl;

    for (long long i = candidate_ref_start; i <= candidate_ref_end; i++) {
        if(i==candidate_ref_start) cout<<"REF:\t";
        cout << "  " << reference_sequence[i - ref_start] << "\t";
        if (max_observed_insert[i - ref_start] > 0) {
            for (uint64_t ii = 0; ii < max_observed_insert[i - ref_start]; ii++)cout << "  *" << "\t";
        }
    }
    cout << endl;


    if(train_mode) {

        for (long long i = candidate_ref_start; i <= candidate_ref_end; i++) {
            if(i==candidate_ref_start) cout<<"TRH1:\t";
            cout << "  " << labels_hp1[i - ref_start] << "\t";
            if (max_observed_insert[i - ref_start] > 0) {
                for (uint64_t ii = 0; ii < max_observed_insert[i - ref_start]; ii++)cout << "  *" << "\t";
            }
        }
        cout << endl;

        for (long long i = candidate_ref_start; i <= candidate_ref_end; i++) {
            if(i==candidate_ref_start) cout<<"TRH2:\t";
            cout << "  " << labels_hp2[i - ref_start] << "\t";
            if (max_observed_insert[i - ref_start] > 0) {
                for (uint64_t ii = 0; ii < max_observed_insert[i - ref_start]; ii++)cout << "  *" << "\t";
            }
        }
        cout << endl;

        for (long long i = candidate_ref_start; i <= candidate_ref_end; i++) {
            if(i==candidate_ref_start) cout<<"TRT1:\t";
            cout << "  " << variant_type_labels_hp1[i - ref_start] << "\t";
            if (max_observed_insert[i - ref_start] > 0) {
                for (uint64_t ii = 0; ii < max_observed_insert[i - ref_start]; ii++)cout << "  *" << "\t";
            }
        }
        cout << endl;

        for (long long i = candidate_ref_start; i <= candidate_ref_end; i++) {
            if(i==candidate_ref_start) cout<<"TRT2:\t";
            cout << "  " << variant_type_labels_hp2[i - ref_start] << "\t";
            if (max_observed_insert[i - ref_start] > 0) {
                for (uint64_t ii = 0; ii < max_observed_insert[i - ref_start]; ii++)cout << "  *" << "\t";
            }
        }
        cout << endl;
    }

    int image_size = candidate.image_matrix[0].size();
    for (int i = 0; i < image_size; i++) {
        cout<< ImageOptionsRegion::column_values[i] <<"\t";
        int region_size = candidate.image_matrix.size();

        for (int j = 0; j < region_size; j++) {
            printf("%3d\t", candidate.image_matrix[j][i]);
        }
        cout << endl;
    }

}