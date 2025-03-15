SNPpath="../../data/PRS/Field22418/merged_chromosome"
filepath="../../data/PRS/ChenGWAS/"
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

tempbedpath=$filepath"temp/temp.bed"
bedpath=$filepath"temp/$file.bed"
splitpath=$filepath"temp/split$file.bed"

Rscript ChenGWAS_step1.R \
--path=$filepath \
--file=$line

awk -F '[:_\t]' '{print $1 "\t" $2 "\t" $5 "\t" $6 "\t" $7 "\t" $8}' $filepath"temp/$file" > $filepath"temp/temp"
sort -k1,1 -k2,2V $filepath"temp/temp" > $filepath"temp/sort$file.bed"
awk -vOFS="\t" '{ printf "%s\t%d\t%d\n", $1, ($2 - 1), $2; }' $filepath"temp/temp" | sort-bed - > $tempbedpath
bedmap --echo --echo-map-id-uniq --delim '\t' $tempbedpath $snppath > $bedpath
awk -F'\t' '{split($4, arr, ";"); print $1, $2, $3, arr[1]}' $bedpath > $splitpath

outpath=$filepath"temp/$file"

Rscript ChenGWAS_step2.R \
--data=$filepath"temp/sort$file.bed" \
--map=$splitpath \
--out=$outpath

awk '!seen[$4]++' $outpath > removedump3

awk '{print $4,$3}' removedump3 > SNP.pvalue3
plink2 \
  --bfile $SNPpath \
  --score removedump3 4 1 2 \
  --q-score-range Pvalue SNP.pvalue3 min \
  --out $targetpath$file
  
rm -rf $filepath"temp"
rm SNP.pvalue3
rm removedump3
fi
done < $filepath"filelist"
rm $filepath/filelist