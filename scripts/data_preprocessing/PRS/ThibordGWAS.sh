SNPpath="../../../data/PRS/Field22418/merged_chromosome"
filepath="../../../data/PRS/ThibordGWAS/"
targetpath="../../../results/PRS/"

ls $filepath*.gz | xargs -n 1 basename > $filepath"filelist"
while read line 
do
file=$(echo $line | cut -d '.' -f 1)
if [ -e $targetpath$file".P.sscore" ]
then
echo $file" Finished!"
else
if [ -e $targetpath$file".P.sscore" ]
then
echo $file" Finished!"
else
mkdir $filepath"temp"

Rscript ThibordGWAS.R \
--path=$filepath \
--file=$line

awk '!seen[$1]++' $filepath"temp/"$file > removedump2
awk '{print $1,$5}' removedump2 > SNP.pvalue2

plink2 \
  --bfile $SNPpath \
  --score removedump2 1 2 4 \
  --q-score-range Pvalue SNP.pvalue2 min \
  --out $targetpath$file
rm -rf $filepath"temp"
rm removedump2
rm SNP.pvalue2
fi
fi
done < $filepath"filelist"
rm $filepath/filelist