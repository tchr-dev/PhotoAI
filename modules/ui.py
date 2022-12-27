import gradio as gr



def upload_file(file):
    file_path = file.name
    return file_path

def build_ui(fn):
    with gr.Blocks() as application:
        with gr.Row():
            with gr.Column():
                image = gr.Image(label='Source image', interactive=True)
                upload_image_btn = gr.UploadButton("Upload image", file_types=["image"], file_count="single")
                upload_image_btn.upload(fn=upload_file, inputs=[upload_image_btn], outputs=[image])
                video = gr.Video(label='Driving video', interactive=True)
                upload_video_btn = gr.UploadButton("Upload video", file_types=["video"], file_count="single")
                upload_video_btn.upload(fn=upload_file, inputs=[upload_video_btn], outputs=[video])
                output_video = gr.Video(label='Output video')
                convert_btn = gr.Button("Convert to video")                
                label_image_params = gr.Label(label="Image params")
                
                convert_btn.click(
                    fn, 
                    inputs=[image, video], 
                    outputs=[output_video])
                    
        return application

