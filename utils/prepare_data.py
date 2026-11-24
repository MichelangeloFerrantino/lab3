import os
import shutil
import subprocess
import zipfile


def prepare_tiny_imagenet(target_dir="data/tiny-imagenet"):
    dataset_dir = os.path.join(target_dir, "tiny-imagenet-200")
    val_dir = os.path.join(dataset_dir, "val")
    val_images_dir = os.path.join(val_dir, "images")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    zip_path = os.path.join(target_dir, "tiny-imagenet-200.zip")

    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(dataset_dir):
        if not os.path.exists(zip_path):
            subprocess.run(
                ["wget", "-q", "http://cs231n.stanford.edu/tiny-imagenet-200.zip", "-O", zip_path],
                check=True,
            )
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

    if os.path.exists(val_images_dir) and os.path.exists(annotations_file):
        with open(annotations_file, "r") as f:
            for line in f:
                fn, cls, *_ = line.split("\t")
                class_dir = os.path.join(val_dir, cls)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(val_images_dir, fn)
                dst = os.path.join(class_dir, fn)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
        shutil.rmtree(val_images_dir)

    return dataset_dir