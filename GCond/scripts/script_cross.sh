dataset=${1}
r=${2}
gpu_id=${3}
for s in 0 1 2 3 4
#对每一次数据集进行5次test save_ours里面有不同seed的结果 每一次的稀疏度会不一样
do
python test_other_arcs.py --dataset ${dataset} --gpu_id=${gpu_id} --r=${r} --seed=${s} --nruns=10  >> res/flickr/${1}_${2}.out 
done
