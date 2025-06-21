# Data Engineering

## Generate each csv dataset from JSON
Example.Dataset.gen_bible("nasb")
Example.Dataset.gen_bible("niv")
Example.Dataset.gen_bible("esv")

## clean up the nasb csv and then niv & esv
%s/\[\([^]]*\)\]/\1/g
%s/’/'/g
%s/‘/'/g

## generate the context len 100 chunks for all 3
Example.Prep.generate("nasb", encoder: true)
Example.Prep.generate("niv", encoder: true)
Example.Prep.generate("esv", encoder: true)

## combine them all into a single csv
mkdir combined
cd combined
cp ../nasbtraining.csv .
cp ../nivtraining.csv .
cp ../esvtraining.csv .
cat *.csv > pretraining.csv

## shuffle and uniq the pretraining data
Example.Prep.shuffle()

# Training

Example.Encoder.scheduled(70, 0.00006)
