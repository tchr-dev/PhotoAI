eval "$(conda shell.bash hook)"
conda activate tpsm

conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
# conda env config vars set CUDA_VISIBLE_DEVICES=0 

conda deactivate

# reactivate
conda activate tpsm

# python demo.py --cpu --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./source.png --driving_video ./driving.mp4
python demo.py --mps --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./Elena.jpg --driving_video ./driving.mp4

conda deactivate

