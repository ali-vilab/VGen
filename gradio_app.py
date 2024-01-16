import os
os.system('pip install "modelscope" --upgrade -f https://pypi.org/project/modelscope/')
os.system('pip install "gradio==3.39.0"')

import gradio as gr
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

image_to_video_pipe = pipeline(task="image-to-video", model='damo/i2vgen-xl', model_revision='v1.1.3', device='cuda:0')

def upload_file(file):
    return file.name

def image_to_video(image_in, text_in):
    if image_in is None:
        raise gr.Error('请上传图片或等待图片上传完成')
    print(image_in)
    output_video_path = image_to_video_pipe(image_in, caption=text_in)[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    return output_video_path


with gr.Blocks() as demo:
    gr.Markdown(
        """<center><font size=7>I2VGen-XL</center>
        <left><font size=3>I2VGen-XL可以根据用户输入的静态图像和文本生成目标接近、语义相同的视频，生成的视频具高清(1280 * 720)、宽屏(16:9)、时序连贯、质感好等特点。</left>

        <left><font size=3>I2VGen-XL can generate videos with similar contents and semantics based on user input static images and text. The generated videos have characteristics such as high-definition (1280 * 720), widescreen (16:9), coherent timing, and good texture.</left>
        """
    )
    with gr.Box():
        gr.Markdown(
        """<left><font size=3>选择合适的图片进行上传，并补充对视频内容的英文文本描述，然后点击“生成视频”。</left>

        <left><font size=3>Please choose the image to upload (we recommend the image size be 1280 * 720), provide the English text description of the video you wish to create, and then click on "Generate Video" to receive the generated video.</left>"""
        )
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="文本描述", lines=2, elem_id="text-in")
                image_in = gr.Image(label="图片输入", type="filepath", interactive=False, elem_id="image-in", height=300)
                with gr.Row():
                    upload_image = gr.UploadButton("上传图片", file_types=["image"], file_count="single")
                    image_submit = gr.Button("生成视频🎬")
            with gr.Column():
                video_out_1 = gr.Video(label='生成的视频', elem_id='video-out_1', interactive=False, height=300)
    gr.Markdown("<left><font size=2>注：如果生成的视频无法播放，请尝试升级浏览器或使用chrome浏览器。</left>")


    upload_image.upload(upload_file, upload_image, image_in, queue=False)
    image_submit.click(fn=image_to_video, inputs=[image_in, text_in], outputs=[video_out_1])

demo.queue(status_update_rate=1, api_open=False).launch(share=False, show_error=True)
