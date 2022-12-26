import gradio as gr
from modules.img2video import convert_image
from argparse import ArgumentParser
import torch
from skimage.transform import resize
from skimage import img_as_ubyte
import imageio.v3 as v3io
import imageio.v2 as v2io

def fn(source_image, driving_video):
    
    # Old code
    
    # Argument's parsing
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='./assets/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='./assets/driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='./result.mp4', help="path to output")
    
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    
    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'], help="Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result")
    
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    
    parser.add_argument("--mps", dest="mps", action="store_true", help="Apple Silicon GPU mode.")

    opt = parser.parse_args()
    
    # ENd Argument parsing
    
    # Forced Arguments Setting
    opt.cpu = True
    opt.mps = False
    
    # End Argument Setting
    return convert_image(
        source_image=source_image, 
        driving_video=driving_video, 
        options=opt)


application = gr.Interface(
    fn, 
    inputs=[gr.Image("./Elena.jpg"), 
            gr.Video("./ds.mp4")], 
    outputs=["text", "playablevideo"],
    cache_examples=False)

if __name__ == "__main__":
    application.launch(share=False)

