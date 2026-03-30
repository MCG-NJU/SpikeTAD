# Test SNN mode on THUMOS14 
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/spiketad/thumos/e2e_thumos_videomae_s_768x1_160_fullft.py --mode snn --checkpoint ./checkpoints/thumos/spiketad_snn_thumos.pth
