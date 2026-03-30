# Test SNN mode on ActivityNet-v1.3
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/spiketad/anet/e2e_anet_videomae_s_192x4_160_fullft.py --mode snn --checkpoint ./checkpoints/anet/spiketad_snn_anet.pth

