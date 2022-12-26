import gradio as gr
from modules.img2video import relative_kp, load_checkpoints, make_animation, find_best_frame
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

    # else:
    #     device = torch.device('mps')
    # Set the device      
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    
    device = torch.device("cpu")
        
    # Load Config

    img = resize(source_image, opt.img_shape)[..., :3]
    video = v3io.imread(driving_video)
    video = [resize(frame, opt.img_shape)[..., :3] for frame in video]
    reader = v2io.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    reader.close()


            
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = opt.config, checkpoint_path = opt.checkpoint, device = device)
 
    # if opt.find_best_frame:
    #     i = find_best_frame(source_image, driving_video, opt.cpu)
    #     print ("Best frame: " + str(i))
    #     driving_forward = driving_video[i:]
    #     driving_backward = driving_video[:(i+1)][::-1]
    #     predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
    #     predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
    #     predictions = predictions_backward[::-1] + predictions_forward[1:]
    # else:
    #     predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
    
    predictions = make_animation(img, video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
    v2io.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    # End of Old code
    
    
    return img.shape, opt.result_video

application = gr.Interface(
    fn, 
    inputs=[gr.Image("./Elena.jpg"), 
            gr.Video("./ds.mp4")], 
    outputs=["text", "playablevideo"],
    cache_examples=False)

if __name__ == "__main__":
    application.launch(share=False)

