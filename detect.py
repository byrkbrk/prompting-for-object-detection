from argparse import ArgumentParser, ArgumentTypeError
from detection import PromptOWLViT



def parse_comma(s):
    """Returns list of string obtained from given string s"""
    try:
        return list(map(str, s.split(",")))
    except ValueError:
        raise ArgumentTypeError("Tuples must be integers seperated by comma")

def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Detects bounding boxes for given image and text prompt")
    parser.add_argument("image_name", type=str, default=None, 
                        help="Name of the image file that be processed. Note image file must be in 'segmentation-images' directory")
    parser.add_argument("text_prompt", nargs="+", type=parse_comma, default=None, help="Text prompt for the model")
    parser.add_argument("--image_size", nargs="+", type=int, default=[1024, 1024],
                        help="Size (height, width) to which the image be transformed")
    return parser.parse_args()    


if __name__ == "__main__":
    args = parse_arguments()
    print(args.text_prompt)
    PromptOWLViT(args.image_name).detect(args.text_prompt, args.image_size, save=True)