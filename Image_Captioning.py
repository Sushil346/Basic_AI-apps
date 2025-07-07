import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import time

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    if image is None:
        return "Please upload an image to get started"
    
    try:
        start_time = time.time()
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=3)
        
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
        
        return f"**Caption:** {caption}\n\n*Processing time: {processing_time:.2f} seconds*"
    
    except Exception as e:
        return f"**Error:** {str(e)}"

custom_css = """
.gradio-container {
    max-width: 1000px;
    margin: 0 auto;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Image Captioning AI ... ")
    gr.Markdown("Using BLIP for AI-generated captions")
    gr.Markdown("Upload an image and get an AI-generated caption")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                height=400,
                show_label=True,
                # value="https://via.placeholder.com/400x300?text=Upload+Image"
            )
        
        with gr.Column(scale=1):
            output_text = gr.Markdown("Upload an image to get started")
            caption_btn = gr.Button("Generate Caption", variant="primary", size="lg")
            gr.Markdown("**Try these examples:**")
            gr.Examples(
                examples=[
                    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",
                    "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=500",
                    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=500"
                ],
                inputs=image_input
            )


    caption_btn.click(
        fn=generate_caption,
        inputs=image_input,
        outputs=output_text
    )
    
    image_input.change(
        fn=generate_caption,
        inputs=image_input,
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()