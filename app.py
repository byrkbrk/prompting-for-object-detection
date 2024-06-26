import gradio as gr
from detection import PromptOWLViT



def detect_objects(image, text_prompt, prompt_owlvit):
    """Returns bounding-boxes-plotted image"""
    text_prompt = text_prompt.split("\n")
    return prompt_owlvit.plot_bboxes_on_image(
        image, 
        prompt_owlvit.get_predictions(image, text_prompt))


if __name__ == "__main__":
    prompt_owlvit = PromptOWLViT(image_name=None, device="cpu")
    gr_interface = gr.Interface(
        fn=lambda image, prompt, owlvit=prompt_owlvit: detect_objects(image, prompt, owlvit),
        inputs=[gr.Image(type="pil"), gr.Textbox(lines=4, placeholder="jacket\nsmall nose\netc")],
        outputs=gr.Image(type="pil")
    )
    gr_interface.launch()
