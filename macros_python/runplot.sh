#!/bin/zsh

MAILTO=""

#source /accelai/app/rshara01/miniforge3/etc/profile.d/conda.sh
#conda activate base

#conda --version

workdir='/Users/rshara01/WORK/LinacTuning/macros_python'
datadir='/Users/rshara01/Desktop/LINAC_STUDY/Daily_snapshots'
imgdir='/Users/rshara01/WORK/LinacTuning/figs/'

cd $workdir

echo $workdir
#whoami
#klist

#dname=`date +%Y_%m_%d-%H_%M_%S`
dname='2022_05_22-00_00_00'
rname='2022_05_21-01_00_00'

echo $dname

#python analyze_daily_data.py --i $datadir --d $dname > /dev/null 2>&1
python analyze_daily_data.py --i $datadir --d $dname --r $rname

mv *png $imgdir/
cd $imgdir

pwd

ls *png > plotlist.txt

