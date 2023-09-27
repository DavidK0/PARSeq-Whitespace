# This script is used for loading a PARSeq checkpoint (.ckpt) and
#   saving it as a TorchScript model (.pt).
# It takes two arguments, the input file and output file.

import sys, argparse

# This assumes parseq and the folder containing this script are in the same folder
sys.path.append("../parseq")
from strhub.models.utils import load_from_checkpoint

    
# Parse the two arguments
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', help="The model checkpoint.")
parser.add_argument("TorchScript", help="The TorchScript output.")
args, unknown = parser.parse_known_args()
    
# Load the model
model = load_from_checkpoint(args.checkpoint).eval()
model.cuda()

# Save the model
model.to_torchscript(args.TorchScript)