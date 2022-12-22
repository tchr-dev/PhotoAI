eval "$(conda shell.bash hook)"
conda activate photoai

# python demo.py --cpu --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./source.png --driving_video ./driving.mp4
python animate.py --mps --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./Elena.jpg --driving_video ./driving.mp4

conda deactivate

