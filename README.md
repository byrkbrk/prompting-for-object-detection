# Prompting for Object Detection

## Introduction

## Setting Up the Environment

## Prompts

Check it out how to use:

~~~
python3 detect.py -h
~~~

Output:

~~~
Detects bounding boxes for given image and text prompts

positional arguments:
  image_name            Name of the image file that be processed. Note image file must be in 'segmentation-images' directory
  text_prompts          Text prompts for the model

options:
  -h, --help            show this help message and exit
  --image_size IMAGE_SIZE [IMAGE_SIZE ...]
                        Size (height, width) to which the image be transformed
~~~

~~~
python3 detect.py dogs.jpg "jacket" "small nose" --image_size 1024 1024
~~~

<p align="center">
  <img src="files-for-readme/dogs.png" width="45%" />
  <img src="files-for-readme/dogs_boxes_on_image.png" width="45%" />
</p>