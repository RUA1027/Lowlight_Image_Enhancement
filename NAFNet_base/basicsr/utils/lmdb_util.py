# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import sys
from multiprocessing import Pool
from os import path as osp

# OpenCV 兼容：有用户环境出现 AttributeError: module 'cv2' has no attribute 'imread'
# 常见原因：误安装了名为 'cv2' 的占位包而非 'opencv-python(-headless)'；或安装损坏。
# 这里增加软回退到 PIL，以保证 LMDB 构建流程不因图像读取失败而完全终止。
try:  # pragma: no cover - 兼容导入
    import cv2  # type: ignore
    _CV2_IMREAD = hasattr(cv2, 'imread')
    _CV2_IMENCODE = hasattr(cv2, 'imencode') and hasattr(cv2, 'IMWRITE_PNG_COMPRESSION')
except Exception:  # noqa: E722
    cv2 = None  # type: ignore
    _CV2_IMREAD = False
    _CV2_IMENCODE = False

import lmdb
from tqdm import tqdm
import numpy as np
from PIL import Image
import io


def _load_image(path):
    """Load image as numpy array (H,W,C) with possible grayscale expansion.

    优先使用 OpenCV；若缺失则使用 PIL 作为回退。
    """
    if _CV2_IMREAD:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # type: ignore
        if img is not None and img.size != 0:
            return img
    # PIL 回退（包括 OpenCV 读到空的情况）
    try:
        with Image.open(path) as im:
            return np.array(im)
    except Exception as e:  # pragma: no cover
        raise FileNotFoundError(f'Failed to load image "{path}" via cv2 and PIL: {e}')


def _encode_png(img, compress_level):
    """Encode numpy array img into PNG bytes respecting compress_level.

    OpenCV 分支返回 bytes；PIL 回退保持等价。"""
    if _CV2_IMENCODE and cv2 is not None:  # type: ignore
        # cv2.imencode 返回 (success, encoded ndarray)
        ok, arr = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])  # type: ignore
        if not ok:
            raise RuntimeError('cv2.imencode failed to encode image to PNG.')
        return arr.tobytes()
    # PIL 回退
    buf = io.BytesIO()
    pil_img = Image.fromarray(img)
    pil_img.save(buf, format='PNG', compress_level=compress_level)
    return buf.getvalue()


def make_lmdb_from_imgs(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        multiprocessing_read=False,
                        n_thread=40,
                        map_size=None):
    """Make lmdb from images.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
        map_size (int | None): Map size for lmdb env. If None, use the
            estimated size from images. Default: None
    """

    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Total images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    dataset = {}
    shapes = {}
    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        pbar = tqdm(total=len(img_path_list), unit='image')

        def callback(arg):
            """get the image data and update pbar."""
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            pbar.set_description(f'Read {key}')

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                read_img_worker,
                args=(osp.join(data_path, path), key, compress_level),
                callback=callback)
        pool.close()
        pool.join()
        pbar.close()
        print(f'Finish reading {len(img_path_list)} images.')

    # create lmdb environment
    if map_size is None:
        # obtain data size from first successfully decoded image
        data_size_per_img = None
        for cand in img_path_list:
            cand_path = osp.join(data_path, cand)
            try:
                img = _load_image(cand_path)
                img_byte = _encode_png(img, compress_level)
                data_size_per_img = len(img_byte)
                print(f'Data size per image (from sample {cand}) is: {data_size_per_img} bytes after PNG encoding')
                break
            except Exception as e:  # skip corrupt/truncated images
                print(f'[Warn] Skip corrupt image during size estimation: {cand_path} ({e})')
                continue
        if data_size_per_img is None:
            raise RuntimeError('No valid images could be decoded (all corrupt or unreadable). Please regenerate debug PNGs or replace with valid placeholders.')
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=map_size)

    # write data to lmdb
    pbar = tqdm(total=len(img_path_list), unit='chunk')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f'Write {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            try:
                img_byte = dataset[key]
                h, w, c = shapes[key]
            except Exception as e:
                print(f'[Warn] Skip corrupt image (multiprocessing) {path}: {e}')
                continue
        else:
            try:
                _, img_byte, img_shape = read_img_worker(
                    osp.join(data_path, path), key, compress_level)
                h, w, c = img_shape
            except Exception as e:
                print(f'[Warn] Skip corrupt image {path}: {e}')
                continue

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


def read_img_worker(path, key, compress_level):
    """Read image worker.

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """

    img = _load_image(path)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    img_byte = _encode_png(img, compress_level)
    return (key, img_byte, (h, w, c))


class LmdbMaker():
    """LMDB Maker.

    Args:
        lmdb_path (str): Lmdb save path.
        map_size (int): Map size for lmdb env. Default: 1024 ** 4, 1TB.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
    """

    def __init__(self,
                 lmdb_path,
                 map_size=1024**4,
                 batch=5000,
                 compress_level=1):
        if not lmdb_path.endswith('.lmdb'):
            raise ValueError("lmdb_path must end with '.lmdb'.")
        if osp.exists(lmdb_path):
            print(f'Folder {lmdb_path} already exists. Exit.')
            sys.exit(1)

        self.lmdb_path = lmdb_path
        self.batch = batch
        self.compress_level = compress_level
        self.env = lmdb.open(lmdb_path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
        self.counter = 0

    def put(self, img_byte, key, img_shape):
        self.counter += 1
        key_byte = key.encode('ascii')
        self.txn.put(key_byte, img_byte)
        # write meta information
        h, w, c = img_shape
        self.txt_file.write(f'{key}.png ({h},{w},{c}) {self.compress_level}\n')
        if self.counter % self.batch == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        self.txn.commit()
        self.env.close()
        self.txt_file.close()
