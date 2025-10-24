SNPpath="../../data/PRS/Field22418/merged_chromosome"
filepath="../../data/PRS/CancerGWAS/"
snppath="../../data/PRS/hg38.dbSNP150.bed"
targetpath="../../results/PRS/"

ls $filepath*.gz | xargs -n 1 basename > $filepath"filelist"
while read line 
do
file=$(echo $line | rev | cut -d'.' -f2- | rev)
if [ -e $targetpath$file".P.sscore" ]
then
echo $file" Finished!"

else
mkdir $filepath"temp"
Rscript CancerGWAS_step1.R \
--path=$filepath \
--file=$line

tempbedpath=$filepath"temp/temp.bed"
bedpath=$filepath"temp/$file.bed"
splitpath=$filepath"temp/split$file.bed"

sort -k1,1 -k2,2V $filepath"temp/$file" > $filepath"temp/sort$file.bed"
awk -vOFS="\t" '{ print $1, ($2 - 1), $2; }' $filepath"temp/$file" | sort-bed - > $tempbedpath
bedmap --echo --echo-map-id-uniq --delim '\t' $tempbedpath $snppath > $bedpath
awk -F'\t' '{split($4, arr, ";"); print $1, $2, $3, arr[1]}' $bedpath > $splitpath

outpath=$filepath"temp/$file"

Rscript CancerGWAS_step2.R \
--data=$filepath"temp/sort$file.bed" \
--map=$splitpath \
--out=$outpath

awk '!seen[$4]++' $outpath > removedump1

awk '{print $4,$3}' removedump1 > SNP.pvalue1
plink2 \
  --bfile $SNPpath \
  --score removedump1 4 1 2 \
  --q-score-range Pvalue SNP.pvalue1 min \
  --out $targetpath$file
  
rm -rf $filepath"temp"
rm SNP.pvalue1
rm removedump1
fi
done < $filepath"filelist"
rm $filepath/filelist