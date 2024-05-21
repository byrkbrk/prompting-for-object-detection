import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from transformers import pipeline
from ultralytics import SAM



class PromptOWLViT(object):
    def __init__(self, 
                 checkpoint_name="google/owlv2-base-patch16-ensemble",
                 task="zero-shot-object-detection"):
        self.detector = pipeline(model=checkpoint_name, task=task)
        self.sam = SAM("mobile_sam.pt")
    
    def detect(self, image_path, labels, save=False):
        image = self.resize_image(self.read_image(image_path))
        predictions = self.detector(image, labels)
        if save:
            self.plot_bboxes_on_image(image, predictions)
        return predictions
    
    def segment(self, image_path, text_labels):
        predictions = self.detect(image_path, text_labels, True)
        box = list(predictions[0]["box"].values())
        p = self.sam(self.resize_image(self.read_image(image_path)), bboxes=box, labels=[1])
        print(p)

    
    def plot_bboxes_on_image(self, image, predictions):
        """Plots bboxes onto image and saves"""
        draw = ImageDraw.Draw(image)
        for prediction in predictions:
            draw.rectangle(list(prediction["box"].values()), outline="red", width=1)
            draw.text((prediction["box"]["xmin"], prediction["box"]["ymin"]),
                      f"{prediction['label']}: {round(prediction['score'], 2)}",
                      fill="white")
        image.save("boxes_on_image.png")

    def read_image(self, image_path):
        """Reads image as Image object"""
        return Image.open(image_path)
    
    def resize_image(self, image):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((1024, 1024)), 
            transforms.ToPILImage()
            ])(image)
    


if __name__ == "__main__":
    prompt_owlvit = PromptOWLViT()
    image_path = "female.jpg"
    labels = ["hair", "eye"]
    prompt_owlvit.detect(image_path, labels)
    prompt_owlvit.segment(image_path, labels)
