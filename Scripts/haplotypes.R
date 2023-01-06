# Generate haplotype matrix using geneHapR

# Install necessary packages
install.packages('genetics')
BiocManager::install('GenomicRanges')
install.packages('pegas')
BiocManager::install('rtracklayer')
BiocManager::install('trackViewer')
install.packages('stringdist')##
install.packages('geneHapR', repos=NULL, type='source')

# Load necessary packages
library(data.table)
library(geneHapR)

# Read in data
setwd('/mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Training_Data')
bvcf <- import_vcf('5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.vcf')
bbed <- import_bed('5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.plink.bed')
livcf <- import_vcf('5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.vcf')
lbed <- import_bed('5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.plink.bed')
gff <- import_gff('Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3')

