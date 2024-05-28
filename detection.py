import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from transformers import pipeline



class PromptOWLViT(object):
    def __init__(self, 
                 image_name,
                 checkpoint_name="google/owlv2-base-patch16-ensemble",
                 task="zero-shot-object-detection",
                 device=None):
        self.module_dir = os.path.dirname(__file__)
        self.image_name = image_name
        self.device = self.initialize_device(device)
        self.detector = pipeline(model=checkpoint_name, task=task, device=self.device)
        self.create_dirs(self.module_dir)
    
    def detect(self, labels, size=(1024, 1024), save=False):
        """Detects bounding boxes for given image path and text labels"""
        image = self.read_image(os.path.join(self.module_dir, "detection-images", self.image_name), 
                                size=size)
        predictions = self.detector(image, labels)
        print("Number of boxes:", len(predictions))
        if save:
            self.plot_bboxes_on_image(image, predictions)
        return predictions
    
    def plot_bboxes_on_image(self, image, predictions, fpath=None):
        """Plots bboxes onto image and saves"""
        draw = ImageDraw.Draw(image)
        for prediction in predictions:
            draw.rectangle(list(prediction["box"].values()), outline="red", width=1)
            draw.text((prediction["box"]["xmin"], prediction["box"]["ymin"]),
                      f"{prediction['label']}: {round(prediction['score'], 2)}",
                      fill="purple")
        
        if fpath is None:
            fpath = os.path.join(self.module_dir, 
                                 "detected-images", 
                                 os.path.splitext(self.image_name)[0] + "_boxes_on_image.png")
        image.save(fpath)
        return image

    def read_image(self, image_path, size=None):
        """Reads image as Image object"""
        image = Image.open(image_path)
        if size:
            image = self.resize_image(image, size)
        return image
    
    def resize_image(self, image, size):
        """Resizes given image into specified size"""
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(size), 
            transforms.ToPILImage()
            ])(image)
    
    def create_dirs(self, root):
        """Creates directories required for detection"""
        dir_names = ["detection-images", "detected-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)
    
    def initialize_device(self, device):
        """Initializes device based on availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)


if __name__ == "__main__":
    image_name = "decouple_1.jpeg"
    #labels = ["hair", "eye", "head", "nose", "jacket"]
    labels = ["jewellery"]
    preds = PromptOWLViT(image_name).detect(labels, save=True)
    print(preds)
