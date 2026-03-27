import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
from PIL import Image
from utils.augmentations import preprocess

class PeopleDetection(data.Dataset):
    """VOC Detection Dataset Object"""
    def __init__(self, root, image_sets='train', transform=None, target_transform=None):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        
        if os.path.exists(os.path.join(self.root, 'images')):
            self.img_folder = os.path.join(self.root, 'images')
            self.ann_folder = os.path.join(self.root, 'annotations')
        else:
            self.img_folder = self.root
            self.ann_folder = self.root

        self.ids = list()
        for file in os.listdir(self.img_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.ids.append(file)
        
        print(f"[Dataset] Found {len(self.ids)} images in {self.img_folder}")

    def __getitem__(self, index):
        img, target, img_path, h, w = self.pull_item(index)
        return img, target, img_path

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        retries = 0
        while True:
            # Prevent infinite loop if dataset is broken
            if retries > 50: 
                print("[CRITICAL] Checked 50 images and found NO valid objects. Check your XML paths/classes.")
                sys.exit(1)

            img_filename = self.ids[index]
            img_id = os.path.splitext(img_filename)[0]
            img_path = os.path.join(self.img_folder, img_filename)

            try:
                img = Image.open(img_path)
                if img.mode == 'L':
                    img = img.convert('RGB')
            except Exception as e:
                index = random.randrange(0, len(self.ids))
                retries += 1
                continue
            
            width, height = img.size

            xml_path = os.path.join(self.ann_folder, img_id + '.xml')
            if not os.path.exists(xml_path):
                 xml_path = os.path.join(self.img_folder, img_id + '.xml')
            
            bbox_labels = []
            if os.path.exists(xml_path):
                try:
                    target = ET.parse(xml_path).getroot()
                    for obj in target.iter('object'):
                        name = obj.find('name').text.lower().strip()
                        bbox = obj.find('bndbox')
                        
                        xmin = (float(bbox.find('xmin').text) - 1) / width
                        ymin = (float(bbox.find('ymin').text) - 1) / height
                        xmax = (float(bbox.find('xmax').text) - 1) / width
                        ymax = (float(bbox.find('ymax').text) - 1) / height

                        # Match class 'person'
                        label_idx = 1 if name == 'person' else 0 
                        
                        if label_idx == 1:
                            bbox_labels.append([label_idx, xmin, ymin, xmax, ymax])
                except Exception:
                    pass

            # If no people found, try next image
            if self.image_set == 'train' and len(bbox_labels) == 0:
                index = random.randrange(0, len(self.ids))
                retries += 1
                continue 

            # Apply Preprocessing
            try:
                img, sample_labels = preprocess(img, bbox_labels, self.image_set, img_path)
            except Exception:
                index = random.randrange(0, len(self.ids))
                retries += 1
                continue

            sample_labels = np.array(sample_labels)
            
            if len(sample_labels) > 0:
                target = np.hstack((sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))
                break
            else:
                if self.image_set == 'train':
                    index = random.randrange(0, len(self.ids))
                    retries += 1
                    continue
                else:
                    target = np.zeros((0, 5))
                    break

        return torch.from_numpy(img), target, img_path, height, width

def detection_collate(batch):
    targets = []
    imgs = []
    paths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        paths.append(sample[2])
    return torch.stack(imgs, 0), targets, paths