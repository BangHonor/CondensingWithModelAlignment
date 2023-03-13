python -u test_amalgamation.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1 --reduction_rate=0.005 --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=0

# for r in 0.02 0.05 
# do
#     python train_gcond_transduct.py --dataset ogbn-arxiv  --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  --lr_adj=0.01 --r=0.01  --seed=1 --inner=3  --epochs=1000  --save=0 
# done

# for r in  0.0125 0.025 0.05
# do
#     python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=${r} --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=0 
# done
#0.5 1.0 2.0

