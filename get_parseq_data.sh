#!/bin/bash

# This script is used to download and process PARSeq's data.
# For more info, see https://github.com/baudm/parseq/blob/main/Datasets.md

# Function to download files from a Google Drive folder
download_files() {
  local folder_url="$1"
  local dest_dir="$2"

  # Extract folder ID from the URL
  local folder_id=$(echo "$folder_url" | awk -F'/' '{print $NF}')

  # Get file IDs from the Google Drive folder
  local file_ids=$(curl -sc /tmp/gdrive-cookie "https://drive.google.com/drive/folders/$folder_id" | \
                   grep -o 'data-id="[^"]*' | awk -F'"' '{print $2}')

  # Create destination directory if not exists
  mkdir -p "$dest_dir"

  # Download files to destination directory
  for file_id in $file_ids; do
    gdown "https://drive.google.com/uc?export=download&id=$file_id" -O "$dest_dir/$file_id"
  done
}

# Define Google Drive folders and target directories
Main_folder="1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE"
OpenVINO_folder="1ym9F1hvhjf7pXEoPMc9oSUUxLOkxA7f-"
TextOCR_folder="1-7vTiARnaa0bgGOgEvymqTzCU6G-qtWq"

# Download files from each folder
download_files "$Main_folder" "data"
download_files "$OpenVINO_folder" "data/OpenVINO"
download_files "$TextOCR_folder" "data/TextOCR"

# Validate and extract datasets
sha256sum -c sha256sums.txt
for d in train test val MJ_train MJ_test MJ_val ST; do
  unzip ${d}.zip
done

# Validate and extract TextOCR datasets
cd TextOCR; sha256sum -c sha256sums.txt; cd -
mkdir -p train/real/TextOCR
for d in train val; do
  unzip TextOCR/${d}.zip -d train/real/TextOCR
done

# Validate and extract OpenVINO datasets
cd OpenVINO; sha256sum -c sha256sums.txt; cd -
mkdir -p train/real/OpenVINO
for d in train_1 train_2 train_5 train_f validation; do
  unzip OpenVINO/${d}.zip -d train/real/OpenVINO
done
