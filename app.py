import os

import cv2

import gradio as gr
import numpy as np
import sys
import io
import torch


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = io.BytesIO()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(bytes(message, encoding='utf-8'))

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


log = Logger()
sys.stdout = log


def read_logs():
    out = log.log.getvalue().decode()
    if out.count("\n") >= 30:
        log.log = io.BytesIO()
    sys.stdout.flush()
    return out


with gr.Blocks() as app:
    gr.Markdown("""
# HINet (or INR-Harmonization) - A novel image Harmonization method based on Implicit neural Networks
## Harmonize any image you want! Arbitrary resolution, and arbitrary aspect ratio! 
### Official Gradio Demo. See here for [**How to play with this Space**](https://github.com/WindVChen/INR-Harmonization/blob/main/assets/demo.gif)
**Since Gradio Space only support CPU, the speed may kind of slow. You may better download the code to run locally with a GPU.**
<a href="https://huggingface.co/spaces/WindVChen/INR-Harmon?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
<img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a> for no queue on your own hardware.</p>
* Official Repo: [INR-Harmonization](https://github.com/WindVChen/INR-Harmonization)
""")

    valid_checkpoints_dict = {"Resolution_256_iHarmony4": "Resolution_256_iHarmony4.pth",
                              "Resolution_1024_HAdobe5K": "Resolution_1024_HAdobe5K.pth",
                              "Resolution_2048_HAdobe5K": "Resolution_2048_HAdobe5K.pth",
                              "Resolution_RAW_HAdobe5K": "Resolution_RAW_HAdobe5K.pth",
                              "Resolution_RAW_iHarmony4": "Resolution_RAW_iHarmony4.pth"}

    global_state = gr.State({
        'pretrained_weight': valid_checkpoints_dict["Resolution_RAW_iHarmony4"],

    })
    with gr.Row():
        with gr.Column():
            form_composite_image = gr.Image(label='Input Composite image', type='pil').style(height="auto")
            gr.Examples(examples=[os.path.join("demo", i) for i in os.listdir("demo") if "composite" in i],
                        label="Composite Examples", inputs=form_composite_image, cache_examples=False)
        with gr.Column():
            form_mask_image = gr.Image(label='Input Mask image', type='pil', interactive=False).style(
                height="auto")
            gr.Examples(examples=[os.path.join("demo", i) for i in os.listdir("demo") if "mask" in i],
                        label="Mask Examples", inputs=form_mask_image, cache_examples=False)
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=2, min_width=10):
                    gr.Markdown(value='Model Selection', show_label=False)

                with gr.Column(scale=4, min_width=10):
                    form_pretrained_dropdown = gr.Dropdown(
                        choices=list(valid_checkpoints_dict.values()),
                        label="Pretrained Model",
                        value=valid_checkpoints_dict["Resolution_RAW_iHarmony4"],
                        interactive=True
                    )

            with gr.Row():
                with gr.Column(scale=2, min_width=10):
                    gr.Markdown(value='Inference Mode', show_label=False)

                with gr.Column(scale=4, min_width=10):
                    form_inference_mode = gr.Radio(
                        ['Square Image', 'Arbitrary Image'],
                        value='Arbitrary Image',
                        interactive=False,
                        label='Mode',
                    )

            with gr.Row():
                with gr.Column(scale=2, min_width=10):
                    gr.Markdown(value='Split Parameter', show_label=False)

                with gr.Column(scale=4, min_width=10):
                    form_split_res = gr.Slider(
                        minimum=0,
                        maximum=2048,
                        step=128,
                        value=256,
                        interactive=False,
                        label="Split Resolution",
                    )
                    form_split_num = gr.Number(
                        value=8,
                        interactive=False,
                        label="Split Number")
            with gr.Row():
                form_log = gr.Textbox(read_logs, label="Logs", interactive=False, type="text", every=1)

        with gr.Column(scale=4):
            form_harmonized_image = gr.Image(label='Harmonized Result', type='numpy', interactive=False).style(
                height="auto")
            form_start_btn = gr.Button("Start Harmonization", interactive=False)
            form_reset_btn = gr.Button("Reset", interactive=True)
            form_stop_btn = gr.Button("Stop", interactive=True)


    def on_change_form_composite_image(form_composite_image):
        if form_composite_image is None:
            return gr.update(interactive=False, value=None), gr.update(value=None)
        return gr.update(interactive=True), gr.update(value=None)


    def on_change_form_mask_image(form_composite_image, form_mask_image):
        if form_mask_image is None:
            return gr.update(interactive=False), gr.update(
                interactive=False if form_composite_image is None else True), gr.update(interactive=False), gr.update(
                interactive=False), gr.update(interactive=False), gr.update(value=None)

        if form_composite_image.size[:2] != form_mask_image.size[:2]:
            raise gr.Error("Composite image and mask image should have the same resolution!")
        else:
            w, h = form_composite_image.size[:2]
            if h != w or (h % 16 != 0):
                return gr.update(value='Arbitrary Image', interactive=False), gr.update(interactive=True), gr.update(
                    interactive=True), gr.update(interactive=True), gr.update(interactive=False,
                                                                              value=-1), gr.update(value=None)
            else:
                return gr.update(value='Square Image', interactive=True), gr.update(interactive=True), gr.update(
                    interactive=True), gr.update(interactive=False), gr.update(interactive=True,
                                                                               value=h // 16,
                                                                               maximum=h,
                                                                               minimum=h // 16,
                                                                               step=h // 16), gr.update(value=None)


    form_composite_image.change(
        on_change_form_composite_image,
        inputs=[form_composite_image],
        outputs=[form_mask_image, form_harmonized_image]
    )

    form_mask_image.change(
        on_change_form_mask_image,
        inputs=[form_composite_image, form_mask_image],
        outputs=[form_inference_mode, form_mask_image, form_start_btn, form_split_num, form_split_res,
                 form_harmonized_image]
    )


    def on_change_form_split_num(form_composite_image, form_split_num):
        w, h = form_composite_image.size[:2]
        if form_split_num < 1:
            return gr.update(value=1)
        elif form_split_num > min(w, h):
            return gr.update(value=min(w, h))
        else:
            return gr.update(value=form_split_num)


    form_split_num.change(
        on_change_form_split_num,
        inputs=[form_composite_image, form_split_num],
        outputs=[form_split_num]
    )


    def on_change_form_inference_mode(form_inference_mode):
        if form_inference_mode == "Square Image":
            return gr.update(interactive=True), gr.update(interactive=False)
        else:
            return gr.update(interactive=False), gr.update(interactive=True)


    form_inference_mode.change(on_change_form_inference_mode, inputs=[form_inference_mode],
                               outputs=[form_split_res, form_split_num])


    def on_click_form_start_btn(form_composite_image, form_mask_image, form_pretrained_dropdown, form_inference_mode,
                                form_split_res, form_split_num):
        log.log = io.BytesIO()
        if form_inference_mode == "Square Image":
            from efficient_inference_for_square_image import parse_args, main_process, global_state
            global_state[0] = 1

            opt = parse_args()
            opt.transform_mean = [.5, .5, .5]
            opt.transform_var = [.5, .5, .5]
            opt.pretrained = os.path.join("./pretrained_models", form_pretrained_dropdown)
            opt.split_resolution = form_split_res
            opt.save_path = None
            opt.workers = 0
            opt.device = "cpu"

            composite_image = np.asarray(form_composite_image)
            mask = np.asarray(form_mask_image)

            try:
                return cv2.cvtColor(
                    main_process(opt, composite_image=composite_image, mask=mask),
                    cv2.COLOR_BGR2RGB)
            except:
                raise gr.Error("Patches too big. Try to reduce the `split_res`!")

        else:
            from inference_for_arbitrary_resolution_image import parse_args, main_process, global_state
            global_state[0] = 1

            opt = parse_args()
            opt.transform_mean = [.5, .5, .5]
            opt.transform_var = [.5, .5, .5]
            opt.pretrained = os.path.join("./pretrained_models", form_pretrained_dropdown)
            opt.split_num = int(form_split_num)
            opt.save_path = None
            opt.workers = 0
            opt.device = "cpu"

            composite_image = np.asarray(form_composite_image)
            mask = np.asarray(form_mask_image)

            try:
                return cv2.cvtColor(
                    main_process(opt, composite_image=composite_image, mask=mask),
                    cv2.COLOR_BGR2RGB)
            except:
                raise gr.Error("Patches too big. Try to increase the `split_num`!")


    generate = form_start_btn.click(on_click_form_start_btn,
                                    inputs=[form_composite_image, form_mask_image, form_pretrained_dropdown,
                                            form_inference_mode,
                                            form_split_res, form_split_num], outputs=[form_harmonized_image])


    def on_click_form_reset_btn(form_inference_mode):
        if form_inference_mode == "Square Image":
            from efficient_inference_for_square_image import global_state
            global_state[0] = 0
        else:
            from inference_for_arbitrary_resolution_image import global_state
            global_state[0] = 0

        log.log = io.BytesIO()
        return gr.update(value=None), gr.update(value=None, interactive=True), gr.update(value=None,
                                                                                         interactive=False), gr.update(
            interactive=False)


    form_reset_btn.click(on_click_form_reset_btn,
                         inputs=[form_inference_mode],
                         outputs=[form_log, form_composite_image, form_mask_image, form_start_btn], cancels=generate)


    def on_click_form_stop(form_inference_mode):
        if form_inference_mode == "Square Image":
            from efficient_inference_for_square_image import global_state
            global_state[0] = 0
        else:
            from inference_for_arbitrary_resolution_image import global_state
            global_state[0] = 0

        log.log = io.BytesIO()
        return gr.update(value=None), gr.update(value=None, interactive=True), gr.update(value=None,
                                                                                         interactive=False), gr.update(
            interactive=False)


    form_stop_btn.click(on_click_form_stop,
                        inputs=[form_inference_mode],
                        outputs=[form_log, form_composite_image, form_mask_image, form_start_btn], cancels=generate)

    gr.Markdown("""
        ## Quick Start
        1. Select desired `Pretrained Model`.
        2. Select a composite image, and then a mask with the same size.
        3. Select the inference mode (for non-square image, only `Arbitrary Image` support).
        4. Set `Split Resolution` (Patches' resolution) or `Split Number` (How many patches, about N*N) according to the inference mode.
        3. Click `Start` and enjoy it!

        """)
    gr.HTML("""
        <style>
            .container {
                position: absolute;
                height: 50px;
                text-align: center;
                line-height: 50px;
                width: 100%;
            }
        </style>
        <div class="container">
        Gradio demo supported by
        <a href="https://github.com/WindVChen">WindVChen</a>
        </div>
        """)

gr.close_all()

app.queue(concurrency_count=1, max_size=200, api_open=False)

app.launch(show_api=False)
