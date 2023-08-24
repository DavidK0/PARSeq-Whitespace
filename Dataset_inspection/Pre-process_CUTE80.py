# This script pre-process the CUTE80 dataset for use by PARSeq.
# Unfortunately CUTE80 seems to be a dataset for ST detection, not ST recognition
#   and thus does not include any labels

import sys, os
import xml.etree.ElementTree as ET

# Check for useage
if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    print(f"Usage: {sys.argv[0]} <path_to_xml_file>")
    sys.exit()
CUTE80_path = sys.argv[1]

# Load the xml file
xml_file_path = os.path.join(CUTE80_path, "Groundtruth\GroundTruth.xml")
tree = ET.parse(xml_file_path)

# Count the number of instances
num_instances = len(tree.getroot())
print(f"Number of instances: {num_instances}")

# Print each instance
for instance in tree.getroot():
    print(f"{instance[0].text} ", end="")