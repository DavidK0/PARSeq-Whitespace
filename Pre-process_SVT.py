# This script pre-process the Street View Text dataset for use by PARSeq.
# The SVT dataset is used primarily for STD or un-cropped STR.
# This script uses the word-level bounding boxes to crop each of the images in SVT
#   to produce a new dataset (SVT_Cropped) which is better suited for PARSeq.

import sys, os
import xml.etree.ElementTree as ElementTree
from PIL import Image

# Check for useage
if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    print(f"Usage: {sys.argv[0]} <path_to_xml_file>")
    sys.exit()
SVT_path = os.path.join(sys.argv[1], "svt1/")

# Load the xml files
train_xml_path = os.path.join(SVT_path, "train.xml")
test_xml_path = os.path.join(SVT_path, "test.xml")
train_xml_tree = ElementTree.parse(train_xml_path)
train_xml_tree = ElementTree.parse(test_xml_path)

# Make a folder for the new dataset
SVT_Cropped_path = "SVT_Cropped"
if not os.path.exists(SVT_Cropped_path):
    os.makedirs(SVT_Cropped_path)

def CropImage(xml_tree, image_index):
    root = xml_tree.getroot()

    image_elements = root.findall('image')
    if image_index < 0 or image_index >= len(image_elements):
        print("Invalid image index.")
        return

    image_element = image_elements[image_index]

    # Get the 'imageName' element's text
    image_name = image_element.find('imageName').text

    # Load the image using PIL
    img = Image.open(os.path.join(SVT_path, image_name))

    data_instances = []

    # Iterate over each 'taggedRectangle' element under 'taggedRectangles'
    tagged_rectangles = image_element.find('taggedRectangles')
    for i, tagged_rectangle in enumerate(tagged_rectangles.findall('taggedRectangle')):
        x = int(tagged_rectangle.attrib.get('x'))
        y = int(tagged_rectangle.attrib.get('y'))
        width = int(tagged_rectangle.attrib.get('width'))
        height = int(tagged_rectangle.attrib.get('height'))

        # Crop the image
        cropped_img = img.crop((x, y, x + width, y + height))

        # Save the cropped image
        cropped_image_name = f"cropped_{image_index}_{i}.png"
        save_path = os.path.join(SVT_Cropped_path, cropped_image_name)
        cropped_img.save(save_path)

        # Record the data instance
        text_label = tagged_rectangle.find('tag').text
        data_instances.append((save_path, text_label))
    return data_instances

# Crop all the images
data_instances = []
for i in range(len(train_xml_tree.getroot())):
    data_instances.extend(CropImage(train_xml_tree, i))

# Save the training data
with open("SVT_train_data.txt", "w") as output_file:
    output_file.write(f"{SVT_Cropped_path}/")
    output_file.write("\n")
    for image_data in data_instances:
        output_file.write(f"{os.path.split(image_data[0])[1]} {image_data[1]}\n")