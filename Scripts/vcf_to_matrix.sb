#!/bin/bash --login
#SBATCH --array=1#-10
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name fastphase2csv
#SBATCH --output=%x_%j.out

cd /mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Training_Data

##### The code here was submitted via SLURM
# Convert fastPHASE files to CSV
# pipe=/mnt/home/seguraab/Shiu_Lab/Project/Scripts/Data_Processing
# path=/mnt/home/seguraab/Shiu_Lab/Collabs/Maize_GxE_Competition_Data/Training_Data
# file=5_Genotype_Data_All_Years_fastphase.chr-2_genotypes.out
# map=5_Genotype_Data_All_Years.plink.map
# save=5_Genotype_Data_All_Years_fastphase.chr-2_genotypes.csv
# python ${pipe}/fastPHASE_to_csv.py -path $path -file $file -map $map -save $save
# Note: The csv is 4928 x 49229; all the other genotype.out files either have
# no genotypes or are all the same size. I don't trust this output... How can 
# some chromosomes not have any snps. It makes no sense.

##### The code below this line was run on the command line
tassel=/mnt/home/seguraab/Shiu_Lab/Project/External_software/tasseladmin-tassel-5-standalone-8b0f83692ccb
plinkk=/mnt/home/seguraab/Shiu_Lab/Project/External_software

# Convert VCFs to hapmap format
${tassel}/run_pipeline.pl -fork1 \
    -vcf 5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.vcf \
    -export 5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.hmp.txt \
    -exportType Hapmap -Xmx30g -Xms30g
${tassel}/run_pipeline.pl -fork1 \
    -vcf 5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.vcf \
    -export 5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.hmp.txt \
    -exportType Hapmap -Xmx40g -Xms10g

# Generate kinship matrices
${tassel}/run_pipeline.pl \
    -importGuess 5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.hmp.txt \
    -KinshipPlugin -method Centered_IBS -endPlugin \
    -export 5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed_kinship.txt \
    -exportType SqrMatrix -Xmx30g -Xms10g

${tassel}/run_pipeline.pl \
    -importGuess 5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.hmp.txt \
    -KinshipPlugin -method Centered_IBS -endPlugin \
    -export 5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed_kinship.txt \
    -exportType SqrMatrix -Xmx40g -Xms10g

# Convert VCF to PLINK format and recode to genotypes (0, 1, 2; number of minor allele)
vcf=5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.vcf
plnk=5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_BEAGLE_imputed.plink
${plinkk}/plink2 --vcf ${vcf} --make-bed --out ${plnk} # convert to PLINK format
${plinkk}/plink --bfile ${plnk} --recode --out ${plnk} # generate ped + map files
${plinkk}/plink --file ${plnk} --recodeA --out ${plnk} # recode genotypes

vcf=5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.vcf
plnk=5_Genotype_Data_All_Years_maf005_maxmiss090_cleaned_LinkImpute_imputed.plink
${plinkk}/plink2 --vcf ${vcf} --make-bed --out ${plnk}
${plinkk}/plink --bfile ${plnk} --recode --out ${plnk}
${plinkk}/plink --file ${plnk} --recodeA --out ${plnk}

scontrol show job $SLURM_JOB_ID