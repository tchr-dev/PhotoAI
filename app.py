import gradio as gr
from modules.img2video import convert_image
from modules.config import get_commandline_options
from modules.ui import build_ui


def fn(source_image, driving_video):
    options = get_commandline_options()
    # Debug mode ON
    options.cpu = True
    options.mps = False
    # Debug mode OFF
    return convert_image(
        source_image=source_image, 
        driving_video=driving_video,
        options=options)

if __name__ == "__main__":
    
    application = build_ui(fn)
    application.launch(share=False)

