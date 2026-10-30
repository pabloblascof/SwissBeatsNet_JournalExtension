import os
import glob
import shutil

import torch
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import save_image
from PIL import Image, ImageOps


def get_transforms(img_size):
    transform = []
    transform.append(v2.ToTensor())
    transform.append(v2.Resize(img_size))
    transform.append(v2.CenterCrop(img_size))
    # transform.append(v2.ToDtype(torch.float32, scale=True))
    # transform.append(
    #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # )
    transform.append(v2.ToPILImage())
    transform = v2.Compose(transform)

    return transform


if __name__ == "__main__":
    img_size = 512
    dir_base = "/capstor/store/cscs/swissai/a02/health_mm_llm_shared/data/MIMIC-CXR-JPG"
    dir_files = os.path.join(dir_base, "files")
    dir_out = f"/capstor/store/cscs/swissai/a02/health_mm_llm_shared/debug/mimic/files{img_size}v2"
    resize_tfs = get_transforms(img_size)

    patient_groups = [
        str_dir
        for str_dir in os.listdir(dir_files)
        if os.path.isdir(os.path.join(dir_files, str_dir))
    ]
    for str_group in sorted(patient_groups):
        patients = [
            str_p
            for str_p in os.listdir(os.path.join(dir_files, str_group))
            if os.path.isdir(os.path.join(dir_files, str_group, str_p))
        ]
        for str_p in sorted(patients):
            studies = [
                str_s
                for str_s in os.listdir(os.path.join(dir_files, str_group, str_p))
                if os.path.isdir(os.path.join(dir_files, str_group, str_p, str_s))
            ]
            print(str_p)
            for str_s in sorted(studies):
                fn_imgs = glob.glob(
                    os.path.join(dir_files, str_group, str_p, str_s, "*.jpg")
                )
                # create novel study directories
                dir_s_out = os.path.join(dir_out, str_group, str_p, str_s)
                if not os.path.exists(dir_s_out):
                    os.makedirs(dir_s_out)
                # copy study reports to mirror the structure of the original
                # image size
                fn_report = os.path.join(dir_files, str_group, str_p, str_s + ".txt")
                fn_report_out = os.path.join(dir_out, str_group, str_p, str_s + ".txt")
                shutil.copyfile(fn_report, fn_report_out)
                for fn_img in fn_imgs:
                    # resize and save all images
                    str_img = fn_img.split("/")[-1]
                    fn_out = os.path.join(dir_out, str_group, str_p, str_s, str_img)
                    # print(fn_img)
                    if not os.path.exists(fn_out):
                        try:
                            # img = read_image(fn_img)
                            img = Image.open(fn_img).convert("L")
                            img = ImageOps.equalize(img)
                            img_out = resize_tfs(img)
                            img_out.save(fn_out)
                            # print(fn_out)
                        except:
                            print("cannot open file...")
                            pass
                    else:
                        print("resized image already exists..")
