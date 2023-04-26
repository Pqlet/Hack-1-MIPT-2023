"""Custom datasets implementations."""
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from opr.datasets.augmentations import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
)
from opr.datasets.base import BaseDataset


# ВНУТРИ BaseDataset идет считывание датафрейма
class PhystechCampus(BaseDataset):
    """Phystech Campus dataset implementation."""

    valid_subsets = ("train", "val", "test")
    # ADDED modality text
    valid_modalities = ("image", "cloud", "image_back")

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"] = "test",
        modalities: Union[str, Tuple[str, ...]] = ("image", "cloud", "image_back"),
        images_subdir: Optional[Union[str, Path]] = "front_cam",
        images_back_subdir: Optional[Union[str, Path]] = "back_cam",
        mink_quantization_size: Optional[float] = 0.5,
    ) -> None:
        """Phystech Campus dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load.. Defaults to "test".
            modalities (Union[str, Tuple[str, ...]]): List of modalities for which the data should be loaded.
                Defaults to ("image", "cloud").
            images_subdir (Union[str, Path]): Images subdirectory path. Defaults to "front_cam".
            images_back_subdir (Union[str, Path]): Images subdirectory path for "back_cam".
            mink_quantization_size (Optional[float]): The quantization size for point clouds. Defaults to 0.5.

        Raises:
            ValueError: If images_subdir is undefined when "images" in modalities.
        """
        super().__init__(dataset_root, subset, modalities)

        if "cloud" in self.modalities:
            self.clouds_subdir = Path("pcd")

        """
        Path(images_subdir) - images_subdir = "front_cam" by default
        this means by default they use fron_cam only
        """
        # ADDED
        # print(self.modalities)


        if "image" in self.modalities:
            if images_subdir:
                self.images_subdir = Path(images_subdir)
            else:
                raise ValueError(
                    "Given 'images' in 'modalities' argument, but 'images_subdir' is set to None"
                )
        if "image_back" in self.modalities:
            if images_back_subdir:
                self.images_back_subdir = Path(images_back_subdir)
                print(self.images_back_subdir)
            else:
                raise ValueError(
                    "Given 'images' in 'modalities' argument, but 'images_back_subdir' is set to None"
                )
        self.clouds_subdir = Path("lidar")

        self.mink_quantization_size = mink_quantization_size

        # TODO workaround to make it compatible with test function code
        if self.subset == "test":
            self.dataset_df["in_query"] = True

        self.image_transform = DefaultImageTransform(train=(subset == "train"), resize=(320, 192))
        self.cloud_transform = DefaultCloudTransform(train=(subset == "train"))
        self.cloud_set_transform = DefaultCloudSetTransform(train=(subset == "train"))

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        data: Dict[str, Union[int, Tensor]] = {"idx": idx}
        row = self.dataset_df.iloc[idx]
        data["utm"] = torch.tensor(row[["northing", "easting"]].to_numpy(dtype=np.float32))
        track_dir = self.dataset_root / str(row["track"])
        if "image" in self.modalities and self.images_subdir is not None:
            im_filepath = track_dir / self.images_subdir / f"{row[f'{self.images_subdir}_ts']}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image"] = im
        # ADDED image_back
        # the subfolder for the image_back is different too
        # i guess the rest can be left untouched
        if "image_back" in self.modalities and self.images_back_subdir is not None:
            im_filepath = track_dir / self.images_back_subdir / f"{row[f'{self.images_back_subdir}_ts']}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image_back"] = im
        if "cloud" in self.modalities and self.clouds_subdir is not None:
            pc_filepath = track_dir / self.clouds_subdir / f"{row['lidar_ts']}.bin"
            pc = self._load_pc(pc_filepath)
            data["cloud"] = pc
        # # ADDED the case for text
        # if "text" in self.modalities:
        #     # data["back_description"] = row["back_description"]
        #     # data["front_description"] = row["front_description"]
        #     data["text_emb_back"] = self.text_transform(row["back_description"])
        #     data["text_emb_front"] = self.text_transform(row["front_description"])

        return data

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))[:, :-1]
        in_range_idx = np.all(
            np.logical_and(-100 <= pc, pc <= 100),  # select points in range [-100, 100] meters
            axis=1,
        )
        pc = pc[in_range_idx]
        pc_tensor = torch.tensor(pc, dtype=torch.float32)
        return pc_tensor
