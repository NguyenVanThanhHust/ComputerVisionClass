import os
import argparse
import shutil
import cv2
import numpy as np

def load_chair_attribute(filename):
    # [1, 3, 4]: Arrays of chair instances
    instance_ids = set()
    with open(filename) as f:
        for line in f.readlines():
            infos = line.split(' # ')
            instance_id = int(infos[0].strip())
            category = infos[4].strip()
            if category == "chair":
                instance_ids.add(instance_id)
    return list(instance_ids)

def convert_segmentation(seg_filename, instance_ids):
    seg_img = cv2.imread(seg_filename)
    unique_instances = np.unique(seg_img[:,:,0])
    seg_out = np.zeros((seg_img.shape[0], seg_img.shape[1]), dtype=np.uint8)
    for idx, instance_id in enumerate(instance_ids):
        try:
            seg_out[seg_img[:, :, 0] == unique_instances[instance_id]] = idx + 1
        except Exception as e:
            print("Error", e)
    return seg_out

def main(data_path, out, **args):
    for subdir1 in os.listdir(data_path):
        input_dir1 = os.path.join(data_path, subdir1)
        if not os.path.isdir(input_dir1): continue
        for subdir2 in os.listdir(input_dir1):
            input_dir2 = os.path.join(input_dir1, subdir2)
            if not os.path.isdir(input_dir2): continue
            for img_name in os.listdir(input_dir2):
                if img_name.startswith(".") or not img_name.endswith(".jpg"): continue
                
                # Make output directory
                out_dir = os.path.join(out, subdir1, subdir2)
                os.makedirs(out_dir, exist_ok=True)
                # Process image
                ## Copy output
                
                ## Process segmentation and save
                instance_ids = load_chair_attribute(os.path.join(input_dir2, img_name.replace(".jpg", "_atr.txt")))
                if len(instance_ids) > 0:
                    shutil.copy(os.path.join(input_dir2, img_name), os.path.join(out_dir, img_name))
                    converted_seg = convert_segmentation(os.path.join(input_dir2, img_name.replace(".jpg", "_seg.png")), instance_ids)
                    cv2.imwrite(os.path.join(out_dir, img_name.replace(".jpg", "_seg.png")), converted_seg)
                    print("Converted {} with {} chairs".format(img_name, len(instance_ids)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ADE20K Chair Category Conversion')
    parser.add_argument('--data_path', type=str, help='path ADE20K/images/training or ADE20K/images/validation')
    parser.add_argument('--out', type=str, help='path ADE20K/images/training or ADE20K/images/validation')
    args = parser.parse_args()
    main(**vars(args))
