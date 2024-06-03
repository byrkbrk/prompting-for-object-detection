from argparse import ArgumentParser
from detection import PromptOWLViT



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Detects bounding boxes for given image and text prompts")
    parser.add_argument("image_name", type=str, default=None, 
                        help="Name of the image file that be processed. Note image file must be in 'segmentation-images' directory")
    parser.add_argument("text_prompts", nargs="+", type=str, default=None, help="Text prompts for the model")
    parser.add_argument("--image_size", nargs="+", type=int, default=[1024, 1024],
                        help="Size (height, width) to which the image be transformed")
    parser.add_argument("--device", type=str, default=None, help="Device that be used during inference")
    return parser.parse_args()    


if __name__ == "__main__":
    args = parse_arguments()
    print("Provided text prompts:", args.text_prompts)
    PromptOWLViT(args.image_name, device=args.device).detect(args.text_prompts, args.image_size, save=True)