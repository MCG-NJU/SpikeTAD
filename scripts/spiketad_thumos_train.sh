# Train on THUMOS14
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/spiketad/thumos/e2e_thumos_videomae_s_768x1_160_fullft.py 

# Test ANN mode on THUMOS14
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/spiketad/thumos/e2e_thumos_videomae_s_768x1_160_fullft.py --mode ann --checkpoint ./exps/thumos/spiketad/gpu4_id0/checkpoint/epoch_59.pth

# Transfer ANN mode to SNN mode and save chekpoints
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/spiketad/thumos/e2e_thumos_videomae_s_768x1_160_fullft.py --mode ann2snn --checkpoint ./exps/thumos/spiketad/gpu4_id0/checkpoint/epoch_59.pth

# Test SNN mode on THUMOS14
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/spiketad/thumos/e2e_thumos_videomae_s_768x1_160_fullft.py --mode snn --checkpoint ./ann2snn_checkpoints/ann2snn.pth
