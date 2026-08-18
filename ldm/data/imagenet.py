import os, yaml, pickle, shutil, tarfile, glob, json, re
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import random
import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
# from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
from saicinpainting.training.data.masks import MixedMaskGenerator
from saicinpainting.evaluation.utils import load_yaml

import torch

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        # example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        try:
            example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        except:
            all = self.labels["file_path_"]
            print(f"2: the length is {len(all)}, the error i is {i}.")
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]


class ImageNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i >= len(self.data):
            i = i % len(self.data)
        try:
            return self.data[i]
        except:
            print(f"0: the length is {len(self.data)}, the error i is {i}.")

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        if (not os.path.exists(self.human2integer)):
            download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        if not self.keep_orig_class_label:
            self.class_labels = [class_dict[s] for s in self.synsets]
        else:
            self.class_labels = [self.synset2idx[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = self.abspaths


class ImageNetTrain(ImageNetBase):
    # NAME = "ILSVRC2012_train"
    NAME = "train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            # cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            # cachedir = "./datasets/ImageNet/"
            cachedir = "/mnt/datasets/"
            self.root = os.path.join(cachedir, self.NAME)

        # self.datadir = os.path.join(self.root, "data")
        self.datadir = self.root
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 1281167
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                    default=True)
        if not tdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                print("Extracting sub-tars.")
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                for subpath in tqdm(subpaths):
                    subdir = subpath[:-len(".tar")]
                    os.makedirs(subdir, exist_ok=True)
                    with tarfile.open(subpath, "r:") as tar:
                        tar.extractall(path=subdir)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.root)


class ImageNetValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            # cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            cachedir = "./.cache"
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 50000
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)
        if not tdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if not os.path.exists(vspath) or not os.path.getsize(vspath)==self.SIZES[1]:
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                print("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.root)



class ImageNetSR(Dataset):
    def __init__(self, size=None, erase_image_with_mask=0.5, Cf_r=0.2,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True, mask_mode="random", heat_source_params=None,
                 heat_source_threshold=0.1, ellipse_params=None,
                 image_read_mode="crop", default_caption="SS304 TIG welding heat source image",
                 caption_key="caption"):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop,
            or directly resizes the full image when image_read_mode="resize"
        2.  resizes crop/full image to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.base = self.get_base()
        self.Cf_r = Cf_r
        self.mask_mode = mask_mode
        self.heat_source_params = heat_source_params or {}
        self.heat_source_threshold = heat_source_threshold
        self.ellipse_params = ellipse_params or {}
        self.image_read_mode = image_read_mode
        self.default_caption = default_caption
        self.caption_key = caption_key
        if self.image_read_mode not in ("crop", "resize"):
            raise ValueError(f"Unknown image_read_mode: {self.image_read_mode}")
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.erase_image_with_mask = erase_image_with_mask
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.image_resizer = albumentations.Resize(height=size, width=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            resampling = getattr(PIL.Image, "Resampling", PIL.Image)
            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": resampling.NEAREST,
            "pil_bilinear": resampling.BILINEAR,
            "pil_bicubic": resampling.BICUBIC,
            "pil_box": resampling.BOX,
            "pil_hamming": resampling.HAMMING,
            "pil_lanczos": resampling.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def _make_heat_source_mask(self, height, width):
        params = self.heat_source_params
        q = float(params.get("Q", 1.0))
        f_front = float(params.get("f_front", 0.5))
        f_back = float(params.get("f_back", 0.5))
        alpha_front = float(params.get("alpha_front", 48.0))
        beta_front = float(params.get("beta_front", 24.0))
        alpha_back = float(params.get("alpha_back", 72.0))
        beta_back = float(params.get("beta_back", 32.0))
        center_x = float(params.get("center_x", 0.5)) * (width - 1)
        center_y = float(params.get("center_y", 0.5)) * (height - 1)
        if params.get("random_angle", False):
            angle_min = float(params.get("angle_min_deg", -45.0))
            angle_max = float(params.get("angle_max_deg", 45.0))
            angle_deg = np.random.uniform(angle_min, angle_max)
        else:
            angle_deg = float(params.get("angle_deg", 0.0))
        angle = np.deg2rad(angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        yy, xx = np.mgrid[0:height, 0:width]
        x = xx.astype(np.float32) - center_x
        y = yy.astype(np.float32) - center_y
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a

        front = (q * f_front / (np.pi * alpha_front * beta_front)) * np.exp(
            -3.0 * ((x_rot ** 2) / (alpha_front ** 2) + (y_rot ** 2) / (beta_front ** 2))
        )
        back = (q * f_back / (np.pi * alpha_back * beta_back)) * np.exp(
            -3.0 * ((x_rot ** 2) / (alpha_back ** 2) + (y_rot ** 2) / (beta_back ** 2))
        )
        mask = front + back
        mask = mask / max(float(mask.max()), 1e-8)

        if self.heat_source_threshold is not None:
            mask = (mask >= float(self.heat_source_threshold)).astype(np.float32)
        return mask.astype(np.float32)

    def _make_random_mask(self, image):
        config = load_yaml("./configs/autoencoder/random_thick_256.yaml")
        mask_kwargs = dict(config.mask_generator_kwargs)
        variants_n = mask_kwargs.pop('variants_n', 2)
        min_mask_area = mask_kwargs.pop('min_mask_area', 0.0)
        max_mask_area = mask_kwargs.pop('max_mask_area', 1.0)
        max_attempts = mask_kwargs.pop('max_attempts', 10)
        mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**mask_kwargs),
                                              variants_n=variants_n)
        fallback = None
        for _ in range(max_attempts):
            src_masks = mask_generator.get_masks(image)
            mask_one_indx = np.random.choice(len(src_masks), 1)
            mask = src_masks[mask_one_indx[0]].astype(np.float32)
            fallback = mask
            mask_area = float(mask.mean())
            if min_mask_area <= mask_area <= max_mask_area:
                return mask
        return fallback.astype(np.float32)

    def _make_ellipse_mask(self, height, width):
        params = self.ellipse_params
        min_radius_x = float(params.get("min_radius_x", 42.0))
        max_radius_x = float(params.get("max_radius_x", 110.0))
        min_radius_y = float(params.get("min_radius_y", 28.0))
        max_radius_y = float(params.get("max_radius_y", 88.0))
        random_angle = bool(params.get("random_angle", True))
        angle_min = float(params.get("angle_min_deg", 0.0))
        angle_max = float(params.get("angle_max_deg", 360.0))

        if params.get("random_center", True):
            margin_x = max_radius_x / max(width - 1, 1)
            margin_y = max_radius_y / max(height - 1, 1)
            center_x = np.random.uniform(margin_x, 1.0 - margin_x) * (width - 1)
            center_y = np.random.uniform(margin_y, 1.0 - margin_y) * (height - 1)
        else:
            center_x = float(params.get("center_x", 0.5)) * (width - 1)
            center_y = float(params.get("center_y", 0.5)) * (height - 1)

        radius_x = np.random.uniform(min_radius_x, max_radius_x)
        radius_y = np.random.uniform(min_radius_y, max_radius_y)
        angle_deg = np.random.uniform(angle_min, angle_max) if random_angle else float(params.get("angle_deg", 0.0))
        angle = np.deg2rad(angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        yy, xx = np.mgrid[0:height, 0:width]
        x = xx.astype(np.float32) - center_x
        y = yy.astype(np.float32) - center_y
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a
        mask = ((x_rot / radius_x) ** 2 + (y_rot / radius_y) ** 2) <= 1.0
        return mask.astype(np.float32)

    def _make_mask(self, image):
        if random.random() < self.Cf_r:
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        if self.mask_mode == "heat_source":
            return self._make_heat_source_mask(image.shape[0], image.shape[1])
        if self.mask_mode == "ellipse":
            return self._make_ellipse_mask(image.shape[0], image.shape[1])
        if self.mask_mode == "random":
            return self._make_random_mask(image)
        raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

    def __getitem__(self, i):
        image = None
        image_path = None
        for offset in range(len(self.base)):
            example = dict(self.base[(i + offset) % len(self.base)])
            image_path = example["file_path_"]
            try:
                with Image.open(image_path) as pil_image:
                    if not pil_image.mode == "RGB":
                        pil_image = pil_image.convert("RGB")
                    else:
                        pil_image.load()
                    image = np.array(pil_image).astype(np.uint8)
                break
            except Exception as exc:
                print(f"Skipping unreadable image: {image_path} ({exc})")
        if image is None:
            raise RuntimeError(f"Failed to load any image after scanning dataset from index {i}")

        if self.image_read_mode == "resize":
            image = self.image_resizer(image=image)["image"]
        else:
            min_side_len = min(image.shape[:2])
            crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
            crop_side_len = int(crop_side_len)

            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
            else:
                self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

            image = self.cropper(image=image)["image"]
            image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]
        # mask generation 掩码生成
        mask = self._make_mask(image)
        image01 = image.astype(np.float32) / 255.0
        masked_image = (1.0 - mask[..., None].astype(np.float32)) * image01
        example["image"] = (image01 * 2.0 - 1.0).astype(np.float32)
        example["masked_image"] = (masked_image * 2.0 - 1.0).astype(np.float32)
        example["LR_image"] = (LR_image / 127.5 - 1.0).astype(np.float32)
        example["mask"] = mask.astype(np.float32)
        example[self.caption_key] = example.get(self.caption_key, self.default_caption)
        return example



class ImageNetSRTrain(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_train_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetTrain(process_images=False,)
        print(f"The original dataset length is: {len(dset)}")
        return Subset(dset, indices)


class ImageNetSRValidation(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_val_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetValidation(process_images=False,)
        return Subset(dset, indices)


class FolderImageDataset(Dataset):
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, data_root, default_caption="SS304 TIG welding heat source image",
                 caption_from_filename=False, caption_file=None,
                 parameter_normalizer=(22.0, 350.0, 75.0, 40.0),
                 parameter_defaults=(0.0, 0.0, 0.0, 0.0),
                 numeric_file=None):
        self.data_root = data_root
        self.default_caption = default_caption
        self.caption_from_filename = caption_from_filename
        self.caption_file = caption_file
        self.caption_map = self._load_caption_map(caption_file)
        self.numeric_file = numeric_file
        self.numeric_map = self._load_caption_map(numeric_file)
        self.parameter_normalizer = np.asarray(parameter_normalizer, dtype=np.float32)
        self.parameter_defaults = np.asarray(parameter_defaults, dtype=np.float32)
        self.paths = sorted(
            path for path in glob.glob(os.path.join(data_root, "**", "*"), recursive=True)
            if os.path.isfile(path) and path.lower().endswith(self.IMG_EXTENSIONS)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {data_root}")
        if self.caption_file is not None:
            print(f"Loaded {len(self.caption_map)} captions from {self.caption_file}")
        if self.numeric_file is not None:
            print(f"Loaded {len(self.numeric_map)} numeric captions from {self.numeric_file}")

    @staticmethod
    def _normalize_key(value):
        key = str(value).replace("\\", "/").strip().lstrip("./")
        key = key.replace("welding speed", "w.s.")
        key = key.replace("welding_speed", "w.s.")
        key = key.replace("cm.m", "cm.m")
        return key

    def _load_caption_map(self, caption_file):
        if caption_file is None:
            return {}
        caption_file = os.path.expanduser(str(caption_file))
        if os.path.isdir(caption_file):
            jsons = sorted(glob.glob(os.path.join(caption_file, "*.json")))
            if len(jsons) == 0:
                raise FileNotFoundError(f"No JSON caption files found under {caption_file}")
            caption_file = jsons[0]
        if not os.path.isfile(caption_file):
            raise FileNotFoundError(f"Caption file not found: {caption_file}")
        self.caption_file = caption_file
        with open(caption_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        caption_map = {}
        if isinstance(data, dict):
            pairs = data.items()
        elif isinstance(data, list):
            pairs = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                key = item.get("file") or item.get("path") or item.get("image") or item.get("image_path")
                caption = item.get("caption") or item.get("prompt") or item.get("text")
                if key is not None and caption is not None:
                    pairs.append((key, caption))
        else:
            pairs = []
        for key, caption in pairs:
            if isinstance(caption, str) and caption.strip():
                caption_map[self._normalize_key(key)] = caption.strip()
        return caption_map

    def _caption_candidates(self, path):
        candidates = []
        relpath = os.path.relpath(path, self.data_root)
        candidates.append(self._normalize_key(relpath))
        candidates.append(self._normalize_key(path))
        candidates.append(os.path.basename(path))
        candidates.append(os.path.splitext(os.path.basename(path))[0])
        seen = set()
        unique = []
        for candidate in candidates:
            candidate = self._normalize_key(candidate)
            if candidate and candidate not in seen:
                unique.append(candidate)
                seen.add(candidate)
        return unique

    def _caption_from_map(self, path):
        for candidate in self._caption_candidates(path):
            if candidate in self.caption_map:
                return self.caption_map[candidate]
        path_key = self._normalize_key(path)
        name_key = self._normalize_key(os.path.basename(path))
        for key, caption in self.caption_map.items():
            if path_key.endswith(key) or key.endswith(name_key):
                return caption
        return None

    def _numeric_from_map(self, path):
        for candidate in self._caption_candidates(path):
            if candidate in self.numeric_map:
                return self.numeric_map[candidate]
        path_key = self._normalize_key(path)
        name_key = self._normalize_key(os.path.basename(path))
        for key, caption in self.numeric_map.items():
            if path_key.endswith(key) or key.endswith(name_key):
                return caption
        return None

    def _extract_numeric_params(self, text):
        text = str(text)
        laser_patterns = [
            r"power\s*[:=]?\s*([-+]?\d*\.?\d+)\s*w",
            r"speed\s*[:=]?\s*([-+]?\d*\.?\d+)\s*mm\s*/?\s*s",
            r"gas\s*flow\s*[:=]?\s*([-+]?\d*\.?\d+)\s*l\s*/?\s*min",
        ]
        laser_matches = [re.search(pattern, text, flags=re.IGNORECASE) for pattern in laser_patterns]
        if laser_matches[0] is not None:
            # Laser welding captions are encoded as [V, I, speed, gas] with V=1 and I=power,
            # so the physical encoder's V * I feature remains the laser power.
            values = [
                1.0,
                float(laser_matches[0].group(1)),
                float(laser_matches[1].group(1)) if laser_matches[1] is not None else float(self.parameter_defaults[2]),
                float(laser_matches[2].group(1)) if laser_matches[2] is not None else float(self.parameter_defaults[3]),
            ]
            raw = np.asarray(values, dtype=np.float32)
            denom = np.maximum(self.parameter_normalizer, 1e-6)
            normalized = raw / denom
            return raw, normalized.astype(np.float32)

        patterns = [
            r"voltage\s*[:=]?\s*([-+]?\d*\.?\d+)\s*v",
            r"current\s*[:=]?\s*([-+]?\d*\.?\d+)\s*a",
            r"speed\s*[:=]?\s*([-+]?\d*\.?\d+)\s*cm\s*/?\s*min",
            r"gas\s*flow\s*[:=]?\s*([-+]?\d*\.?\d+)\s*l\s*/?\s*min",
        ]
        values = []
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is None:
                values.append(float(self.parameter_defaults[idx]))
            else:
                values.append(float(match.group(1)))
        raw = np.asarray(values, dtype=np.float32)
        denom = np.maximum(self.parameter_normalizer, 1e-6)
        normalized = raw / denom
        return raw, normalized.astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        caption_path = os.path.splitext(path)[0] + ".txt"
        caption = self.default_caption
        mapped_caption = self._caption_from_map(path)
        if mapped_caption is not None:
            caption = mapped_caption
        elif os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip() or caption
        elif self.caption_from_filename:
            caption = os.path.splitext(os.path.basename(path))[0].replace("_", " ").replace("-", " ")
        numeric_source = self._numeric_from_map(path) or caption
        numeric_raw, numeric_params = self._extract_numeric_params(numeric_source)
        return {
            "file_path_": path,
            "relpath": os.path.relpath(path, self.data_root),
            "caption": caption,
            "numeric_params": numeric_raw,
            "numeric_params_norm": numeric_params,
            "numeric_params_raw": numeric_raw,
        }


class SS304SRTrain(ImageNetSR):
    def __init__(self, data_root, default_caption="SS304 TIG welding heat source image",
                 caption_from_filename=False, caption_file=None,
                 parameter_normalizer=(22.0, 350.0, 75.0, 40.0),
                 numeric_file=None, **kwargs):
        self.data_root = data_root
        self.default_caption = default_caption
        self.caption_from_filename = caption_from_filename
        self.caption_file = caption_file
        self.parameter_normalizer = parameter_normalizer
        self.numeric_file = numeric_file
        super().__init__(**kwargs)

    def get_base(self):
        dset = FolderImageDataset(
            self.data_root,
            default_caption=self.default_caption,
            caption_from_filename=self.caption_from_filename,
            caption_file=self.caption_file,
            parameter_normalizer=self.parameter_normalizer,
            numeric_file=self.numeric_file,
        )
        print(f"The SS304 train dataset length is: {len(dset)}")
        return dset


class SS304SRValidation(ImageNetSR):
    def __init__(self, data_root, default_caption="SS304 TIG welding heat source image",
                 caption_from_filename=False, caption_file=None,
                 parameter_normalizer=(22.0, 350.0, 75.0, 40.0),
                 numeric_file=None, **kwargs):
        self.data_root = data_root
        self.default_caption = default_caption
        self.caption_from_filename = caption_from_filename
        self.caption_file = caption_file
        self.parameter_normalizer = parameter_normalizer
        self.numeric_file = numeric_file
        super().__init__(random_crop=False, **kwargs)

    def get_base(self):
        dset = FolderImageDataset(
            self.data_root,
            default_caption=self.default_caption,
            caption_from_filename=self.caption_from_filename,
            caption_file=self.caption_file,
            parameter_normalizer=self.parameter_normalizer,
            numeric_file=self.numeric_file,
        )
        print(f"The SS304 validation dataset length is: {len(dset)}")
        return dset


class LaserWeldingSRTrain(ImageNetSR):
    def __init__(self, data_root, default_caption="laser welding image",
                 caption_from_filename=False, caption_file=None, label_json=None,
                 parameter_normalizer=(1.0, 2400.0, 30.0, 25.0),
                 numeric_file=None, **kwargs):
        self.data_root = data_root
        self.default_caption = default_caption
        self.caption_from_filename = caption_from_filename
        self.caption_file = caption_file if caption_file is not None else label_json
        self.parameter_normalizer = parameter_normalizer
        self.numeric_file = numeric_file
        super().__init__(**kwargs)

    def get_base(self):
        dset = FolderImageDataset(
            self.data_root,
            default_caption=self.default_caption,
            caption_from_filename=self.caption_from_filename,
            caption_file=self.caption_file,
            parameter_normalizer=self.parameter_normalizer,
            numeric_file=self.numeric_file,
        )
        print(f"The laserwelding train dataset length is: {len(dset)}")
        return dset


class LaserWeldingSRValidation(ImageNetSR):
    def __init__(self, data_root, default_caption="laser welding image",
                 caption_from_filename=False, caption_file=None, label_json=None,
                 parameter_normalizer=(1.0, 2400.0, 30.0, 25.0),
                 numeric_file=None, **kwargs):
        self.data_root = data_root
        self.default_caption = default_caption
        self.caption_from_filename = caption_from_filename
        self.caption_file = caption_file if caption_file is not None else label_json
        self.parameter_normalizer = parameter_normalizer
        self.numeric_file = numeric_file
        super().__init__(random_crop=False, **kwargs)

    def get_base(self):
        dset = FolderImageDataset(
            self.data_root,
            default_caption=self.default_caption,
            caption_from_filename=self.caption_from_filename,
            caption_file=self.caption_file,
            parameter_normalizer=self.parameter_normalizer,
            numeric_file=self.numeric_file,
        )
        print(f"The laserwelding validation dataset length is: {len(dset)}")
        return dset

class ImageNetinfer(Dataset):
    def __init__(self, indir="your val data dir"):
        self.masks = sorted(glob.glob(os.path.join(indir, "*_mask.png")))
        self.images = [x.replace("_mask.png", ".png") for x in self.masks]

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, i):
        # cv2.setNumThreads(0)
        image = self.images[i]
        mask = self.masks[i]
        image_dir = image
        image = np.array(Image.open(image).convert("RGB"))
        image = image.astype(np.float32) / 255.0

        mask = np.array(Image.open(mask).convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        masked_image = (1 - mask[:,:,None]) * image
        example = dict()
        example["image"] = image * 2.0 - 1.0
        example["masked_image"] = masked_image * 2.0 - 1.0
        example["mask"] = mask * 2.0 - 1.0
        example["image_dir"] = image_dir
        return example

class ImageNetsv(Dataset): # This part is not public
    def __init__(self, indir="your val data dir"):
        self.sampler = range(0,50)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, i):
        return i
