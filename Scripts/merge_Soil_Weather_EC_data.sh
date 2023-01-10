#! /bin/sh

#replace the num by PCAnumber
for i in `seq 0 10`; do echo "sed -i '1 s/$i/ECPCA$i/' 6_Training_EC_centered_PCA.csv"; done > flow.sh
for i in `seq 0 125`; do echo "sed -i '1 s/$i/WPCA$i/' 4_Weather_centered_PCA.csv"; done >> flow.sh
sh flow.sh

#get weather lacking and fill up weather data
awk -F"," '{print $1}' 4_Weather_centered_PCA.csv | sort | uniq | grep -v Env > 4_weather_env
cat Env_namelist 4_weather_env | sort | uniq -u > Weather_lacking_namelist
for i in `cat Weather_lacking_namelist`; do echo $i,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,; done >> 4_Weather_centered_PCA.csv

#get EC lacking Env and fill up EC lacking data
awk -F"," '{print $1}' 6_Training_EC_centered_PCA.csv | sort | uniq | grep -v Env > 6_EC_env
cat Env_namelist 6_EC_env | sort | uniq -u | grep -v Env > EC_lacking_namelist
for i in `cat EC_lacking_namelist`; do echo $i,,,,,,,,,,,; done >> 6_Training_EC_centered_PCA.csv

#soil data
cut -d"," -f1,6-28,30-35 3_Training_Soil_Data_2015_2021_cleaned.csv | sort | uniq > 3_Training_Soil_Data_2015_2021_newdataset.csv
sed -i '2,$ s/"//g' 3_Training_Soil_Data_2015_2021_newdataset.csv
awk -F"," '{print $1}' 3_Training_Soil_Data_2015_2021_newdataset.csv | sort | uniq | grep -v Env > 3_soil_env
cat Env_namelist 3_soil_env | sort | uniq -u | grep -v Env> Soil_lacking_namelist
for i in `cat Soil_lacking_namelist`; do echo $i,,,,,,,,,,,,,,,,,,,,,,,,,,,,,; done >> 3_Training_Soil_Data_2015_2021_newdataset.csv

#merge soil,EC and weather data using Env, remember to move Env to first line
for i in 3_Training_Soil_Data_2015_2021_newdataset.csv 4_Weather_centered_PCA.csv 6_Training_EC_centered_PCA.csv ; do echo $i; sort -k 1 $i > srt_$i; done
join -t"," -1 1 -2 1 srt_3_Training_Soil_Data_2015_2021_newdataset.csv srt_4_Weather_centered_PCA.csv > temp_txt
join -t"," -1 1 -2 1 temp_txt srt_6_Training_EC_centered_PCA.csv > merged_Soil_Weather_EC_All.txt
