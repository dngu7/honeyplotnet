# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import glob
import json
from tqdm.contrib.concurrent import process_map

from .gen_helper import download_file, unzip_file

def download_icpr_dataset(data_dir, data_urls):
  # Download and extract data from https://chartinfo.github.io/
  for _, url in data_urls.items():
    zip_path = os.path.join(data_dir, url.split('/')[-1])
    download_file(url, zip_path)

    extract_path = os.path.join(data_dir, url.split('/')[-1].replace('.zip',''))
    if not os.path.exists(extract_path):
      unzip_file(zip_path, data_dir)

def process_icpr_records(data_dir, icpr_dir):

  icpr_records = {}
  for split, split_dir in icpr_dir.items():
    split_path = os.path.join(data_dir, split_dir)

    if split == 'train':
      icpr_records[split] = process_icpr_train(split_path)
    elif split == 'test':
      icpr_records[split] = process_icpr_eval(split_path)
  
  return icpr_records


def process_icpr_train(train_dir):

  images_dir = os.path.join(train_dir, 'images')
  assert os.path.exists(images_dir), images_dir
  chart_img_dirs = glob.glob(images_dir + '/*')
  
  data = []
  for chart_img_dir in chart_img_dirs:
    chart_jpgs = glob.glob(chart_img_dir + '/*')
    
    #Loop through each image
    for jpg_path in chart_jpgs:
      json_path = jpg_path.replace('/images', '/annotations_JSON').replace('.jpg', '.json')
      assert os.path.exists(json_path), json_path
      
      with open(json_path) as f:
        record = json.load(f)

      record['json_path'] = json_path
      record['jpg_path'] = jpg_path
      data.append(record)
  return data

def process_icpr_eval(test_dir):
  task_to_split = {
      1: 'split_1',
      2: 'split_2',
      3: 'split_3', 
      4: 'split_3',
      5: 'split_3',
      6: 'split_4',
      7: 'split_5'
  }

  test_split_dirs = os.path.join(test_dir, 'splits_with_GT')
  data = []
  for split_name in list(set(list(task_to_split.values()))):

    test_split_dir = os.path.join(test_split_dirs, split_name)
    images_dir = os.path.join(test_split_dir, 'images')
    all_jpgs = glob.glob(images_dir + '/*')
    
    for jpg_path in all_jpgs:
      json_path = jpg_path.replace('/images', '/annotations_JSON').replace('.jpg', '.json')
      assert os.path.exists(json_path), json_path
      with open(json_path) as f:
        record = json.load(f)
      record['json_path'] = json_path
      record['jpg_path'] = jpg_path
      record['split_id'] =  split_name
      data.append(record)

  return data  
