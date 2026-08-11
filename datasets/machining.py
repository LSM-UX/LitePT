import os
from collections.abc import Sequence

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


def _load_array(path, preferred_keys):
    """Load one array from either an NPY file or a simple NPZ archive."""
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            for key in preferred_keys:
                if key in value.files:
                    return value[key]
            if len(value.files) == 1:
                return value[value.files[0]]
            raise KeyError(
                f"Cannot choose an array from {path}; available keys: {value.files}"
            )
        finally:
            value.close()
    return value


@DATASETS.register_module()
class MachiningDataset(DefaultDataset):
    """Semantic-segmentation dataset stored as split/points + split/labels.

    A point file must have shape (N, >=6), with XYZ in columns 0:3 and surface
    normals in columns 3:6.  Its label file must have shape (N,) (or an
    equivalent shape that can be flattened).  NPY and NPZ files are supported.
    """

    def __init__(
        self,
        points_dirname="points",
        labels_dirname="labels",
        max_samples=None,
        **kwargs,
    ):
        self.points_dirname = points_dirname
        self.labels_dirname = labels_dirname
        self.max_samples = max_samples
        super().__init__(**kwargs)

    @staticmethod
    def _file_map(folder):
        result = {}
        for filename in sorted(os.listdir(folder)):
            stem, extension = os.path.splitext(filename)
            if extension.lower() in (".npy", ".npz"):
                if stem in result:
                    raise RuntimeError(f"Duplicate sample stem '{stem}' in {folder}")
                result[stem] = os.path.join(folder, filename)
        return result

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise TypeError(f"split must be a string or sequence, got {type(self.split)}")

        data_list = []
        for split in split_list:
            points_dir = os.path.join(self.data_root, split, self.points_dirname)
            labels_dir = os.path.join(self.data_root, split, self.labels_dirname)
            if not os.path.isdir(points_dir) or not os.path.isdir(labels_dir):
                raise FileNotFoundError(
                    f"Expected point/label folders: {points_dir} and {labels_dir}"
                )

            points = self._file_map(points_dir)
            labels = self._file_map(labels_dir)
            missing_labels = sorted(set(points) - set(labels))
            missing_points = sorted(set(labels) - set(points))
            if missing_labels or missing_points:
                raise RuntimeError(
                    f"Unmatched files in split '{split}': "
                    f"missing labels={missing_labels[:5]}, missing points={missing_points[:5]}"
                )
            data_list.extend((points[name], labels[name], split, name) for name in sorted(points))
        if self.max_samples is not None:
            if self.max_samples <= 0:
                raise ValueError(f"max_samples must be positive, got {self.max_samples}")
            data_list = data_list[: self.max_samples]
        return data_list

    def get_data(self, idx):
        point_path, label_path, split, name = self.data_list[idx % len(self.data_list)]
        points = np.asarray(_load_array(point_path, ("points", "point", "data")))
        labels = np.asarray(_load_array(label_path, ("labels", "label", "segment"))).reshape(-1)

        if points.ndim != 2 or points.shape[1] < 6:
            raise ValueError(f"Expected points with shape (N, >=6), got {points.shape}: {point_path}")
        if points.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Point/label length mismatch ({points.shape[0]} vs {labels.shape[0]}): "
                f"{point_path}, {label_path}"
            )

        return dict(
            coord=points[:, :3].astype(np.float32, copy=True),
            normal=points[:, 3:6].astype(np.float32, copy=True),
            segment=labels.astype(np.int32, copy=False),
            name=name,
            split=split,
        )

    def get_data_name(self, idx):
        return self.data_list[idx % len(self.data_list)][3]

    def get_split_name(self, idx):
        return self.data_list[idx % len(self.data_list)][2]
