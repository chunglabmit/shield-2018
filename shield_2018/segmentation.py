'''segmentation.py - the segmentation method used in the shield paper'''

import argparse
import glob
import itertools
import json
import logging
import multiprocessing
import neuroglancer
import numpy as np
import os
import tifffile
import tqdm
import torch
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_dilation, label, generate_binary_structure
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree
from skimage.morphology import watershed
import sys
try:
    from pathom.segmentation.torch_impl import eigvals_of_weingarten
except ImportError:
    from phathom.segmentation.segmentation import eigvals_of_weingarten
from phathom.utils import SharedMemory

#
# Globals for the analysis
#
x_extent = None
y_extent = None
z_extent = None
block_size_x = None
block_size_y = None
block_size_z = None
padding = None
stack_files = None
#
# Globals for multiprocessing
#
io_threads = 4
processing_threads = multiprocessing.cpu_count()
#
# Parameters for the segmentation
#
"""The minimum allowed threshold for the first Otsu"""
t1min = 1.0
"""The maximum allowed threshold for the first Otsu"""
t1max = 5.0
"""The minimum allowed threshold for the second Otsu"""
t2min = 4.0
"""The maximum allowed threshold for the second Otsu"""
t2max = 10.0
"""The minimum area of a segmented object"""
min_area = 20
"""The sigma for the first gaussian of the difference of gaussians"""
dog_low = 3
"""The sigma for the second gaussian of the difference of gaussians"""
dog_high = 10
#
# These are global shared memory regions for multiprocessing
#
"""SharedMemory for the image stack"""
stackmem = None

"""SharedMemory for the difference of gaussians"""
dogmem = None

"""A normal Numpy array to hold the results of the curvature computation"""
curv = None

def parse_args(args=sys.argv[1:]):
    global padding, t1max, t1min, t2max, t2min, min_area, dog_low, dog_high
    global io_threads, processing_threads
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        required=True,
                        help="A glob expression for the stack of images to "
                      "analyze, for instance, \"/path-to/img_*.tiff\".")
    parser.add_argument("--output",
                        required=True,
                        help="The path to the output json file of cell centers")
    parser.add_argument("--block-size-x",
                        type=int,
                        default=1024,
                        help="The size of a processing block in the X direction")
    parser.add_argument("--block-size-y",
                        type=int,
                        default=1024,
                        help="The size of a processing block in the Y direction")
    parser.add_argument("--block-size-z",
                        type=int,
                        default=200,
                        help="The size of a processing block in the Z direction")
    parser.add_argument("--crop-x",
                        help="Crop the volume to these x coordinates. "
                        "Specify as \"--crop-x <x-min>,<x-max>\".")
    parser.add_argument("--crop-y",
                        help="Crop the volume to these y coordinates. "
                        "Specify as \"--crop-y <y-min>,<y-max>\".")
    parser.add_argument("--crop-z",
                        help="Crop the volume to these z coordinates. "
                        "Specify as \"--crop-z <z-min>,<z-max>\".")
    parser.add_argument("--padding",
                        type=int,
                        default=30,
                        help="The amt of padding to add to a block")
    parser.add_argument("--io-threads",
                        type=int,
                        default=io_threads,
                        help="# of threads to use when reading image files")
    parser.add_argument("--processing-threads",
                        type=int,
                        default=processing_threads,
                        help="# of threads to use during computation")
    t1max, t1min, t2max, t2min, min_area, dog_low, dog_high
    parser.add_argument("--t1min",
                        type=float,
                        default=t1min,
                        help="The minimum allowed threshold for the low Otsu")
    parser.add_argument("--t1max",
                        type=float,
                        default=t1max,
                        help="The maximum allowed threshold for the low Otsu")
    parser.add_argument("--t2min",
                        type=float,
                        default=t2min,
                        help="The minimum allowed threshold for the high Otsu")
    parser.add_argument("--t2max",
                        type=float,
                        default=t2max,
                        help="The maximum allowed threshold for the hig Otsu")
    parser.add_argument("--dog-low",
                        type=float,
                        default=dog_low,
                        help="The sigma for the foreground gaussian for the "
                        "difference of gaussians")
    parser.add_argument("--dog-high",
                        type=float,
                        default=dog_high,
                        help="The sigma for the background gaussian for the "
                        "difference of gaussians")
    parser.add_argument("--log-level",
                        default="INFO",
                        help="The log level for the Python logger: one of "
                        '"DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".')
    parser.add_argument("--log-file",
                        help="File to log to. Default is console.")
    parser.add_argument("--log-format",
                        help="Format for log messages. See "
                        "https://docs.python.org/3/howto/logging.html"
                        "#changing-the-format-of-displayed-messages for help")
    pargs = parser.parse_args(args)
    io_threads = pargs.io_threads
    processing_threads = pargs.processing_threads
    padding = pargs.padding
    t1min = pargs.t1min
    t1max = pargs.t1max
    t2min = pargs.t2min
    t2max = pargs.t2max
    dog_low = pargs.dog_low
    dog_high = pargs.dog_high
    return pargs

def get_padded_coords(x0, x1, y0, y1, z0, z1):
    """Return the block size to use after padding

    :param x0: the minimum x coordinate of the region to analyze
    :param x1: the maximum x coordinate of the region to analyze
    :param y0: the minimum y coordinate of the region to analyze
    :param y1: the maximum y coordinate of the region to analyze
    :param z0: the minimum z coordinate of the region to analyze
    :param z1: the maximum z coordinate of the region to analyze
    :return: a tuple of xmin, xmax, ymin, ymax, zmin and zmax for the padded
    region
    """
    z0a = max(z0 - padding, 0)
    z1a = min(z1 + padding, z_extent)
    y0a = max(y0 - padding, 0)
    y1a = min(y1 + padding, y_extent)
    x0a = max(x0 - padding, 0)
    x1a = min(x1 + padding, x_extent)
    return x0a, x1a, y0a, y1a, z0a, z1a


def read_plane(z, off=0):
    """Read a plane of memory into the image stack memory

    :param z: the Z coordinate of the plane
    :param off: the offset of the first plane in the stackmem
    """
    with stackmem.txn() as m:
        m[z-off] = tifffile.imread(stack_files[z])


def compute_dog(x0, x1, y0, y1, xoff, yoff):
    """Compute the difference of gaussians for the given region

    Using the image in stackmem, compute the difference of gaussians for a chunk

    :param x0: the minimum x of the chunk to do
    :param x1: the maximum x of the chunk to do
    :param y0: the minimum y of the chunk to do
    :param y1: the maximum y of the chunk to do
    :param xoff: the x offset into the DOG memory
    :param yoff: the y offset into the DOG memory
    """
    x0a, x1a, y0a, y1a, z0a, z1a = get_padded_coords(
        x0, x1, y0, y1, 0, dogmem.shape[0])
    with stackmem.txn() as stack:
        img = stack[:, y0a:y1a, x0a:x1a].astype(np.float32)
    dog = gaussian_filter(img, dog_low) - gaussian_filter(img, dog_high)
    with dogmem.txn() as m:
        m[:, y0-yoff:y1-yoff, x0-xoff:x1-xoff] = \
            dog[:, y0-y0a:y1-y0a, x0-x0a:x1-x0a]


def segment(dog, curv, x0, x1, y0, y1, z0, z1):
    """Segment a block

    :param dog: the difference of gaussians for the volume being processed
    :param curv: the curvature for the volume being processed
    :param x0: the block x start
    :param x1: the block x end
    :param y0: the block y start
    :param y1: the block y end
    :param z0: the block z start
    :param z1: the block z end
    :return: an Nx4 numpy array containing the Z, Y, X coordinates and area
    per detected cell
    """
    x0a, x1a, y0a, y1a, z0a, z1a = get_padded_coords(x0, x1, y0, y1, z0, z1)
    dogg = dog[z0a:z1a, y0a:y1a, x0a:x1a]
    curvv = curv[z0a:z1a, y0a:y1a, x0a:x1a]
    curvv[np.isnan(curvv)] = 0
    curvvt1 = curvv[(curvv > 0) & (curvv < 100)]
    if len(curvvt1) == 0:
        return np.zeros((0, 4))
    try:
        t1 = min(t1max, max(t1min, threshold_otsu(curvvt1)))
    except ValueError:
        t1 = t1min
    curvvt2 = curvv[(curvv > t1) & (curvv < 100)]
    if len(curvvt2) == 0:
        t2 = t1
    else:
        try:
            t2 = min(t2max, max(t2min, threshold_otsu(curvvt2)))
        except ValueError:
            t2 = t2min
    tseed = (t1 + t2) / 2
    footprint = np.sum(np.square(np.mgrid[-5:6, -5:6, -5:6]), 0) <= 25
    seed_mask = (curvv > tseed) & (grey_dilation(curvv, footprint=footprint) == curvv)
    seeds, count = label(seed_mask)
    if count == 0:
        return np.zeros((0, 4))
    imask = curvv > t1
    seg = watershed(-curvv, seeds, mask=imask)
    zs, ys, xs = np.where(seg > 0)
    a = np.bincount(seg[zs, ys, xs])[1:]
    zc, yc, xc = [np.bincount(seg[zs, ys, xs], s)[1:] / a + off
                  for s, off in ((zs, z0a), (ys, y0a), (xs, x0a))]
    mask = (zc >= z0) & (zc < z1) & (yc >= y0) & (yc < y1) & (xc >= x0) & (xc < x1)
    return np.column_stack((zc[mask], yc[mask], xc[mask], a[mask]))


def mpsegment(x0, x1, y0, y1, z0, z1):
    global dogmem
    with dogmem.txn() as dog:
        return segment(dog, curv, x0, x1, y0, y1, z0, z1)


def segment_block(x0, x1, y0, y1, z0a, z1a):
    """segment some of the planes in the z-stack

    :param x0:
    :param x1:
    :param y0:
    :param y1:
    :param z0a:
    :param z1a:
    :return: a 2-tuple of an Nx3 array of coordinates of objects and areas of
    those objects
    """
    global dogmem, stackmem, curv
    coords = []
    _, _, _, _, z0p, z1p = get_padded_coords(
        0, x_extent, 0, y_extent, z0a, z1a)
    logging.info("Processing %d to %d" % (z0a, z1a))
    stackmem = SharedMemory((z1p - z0p, y_extent, x_extent), np.uint16)
    with multiprocessing.Pool(io_threads) as pool:
        futures = []
        logging.info("Enqueueing reads")
        for z in range(z0p, z1p):
            futures.append(pool.apply_async(read_plane, (z, z0p)))
        logging.info("Processing %d reads" % len(futures))
        for future in tqdm.tqdm(futures):
            future.get()
    x0p, x1p, y0p, y1p, _, _ = get_padded_coords(x0[0], x1[-1],
                                                 y0[0], y1[-1],
                                                 0, z1p - z0p)
    dogmem = SharedMemory((z1p - z0p, y1p-y0p, x1p-x0p), np.float32)
    futures = []
    with multiprocessing.Pool(processing_threads) as pool:
        for (x0a, x1a), (y0a, y1a) in itertools.product(
                zip(x0, x1), zip(y0, y1)):
            futures.append(pool.apply_async(
                compute_dog, (x0a, x1a, y0a, y1a, x0p, y0p)))
        for future in tqdm.tqdm(futures):
            future.get()
    with dogmem.txn() as dog:
        e = eigvals_of_weingarten(dog[:, :y1p-y0p, :x1p-x0p])
        curv = np.all(e < 0, -1) * e[..., 0] * e[..., 1]
    with multiprocessing.Pool(processing_threads) as pool:
        futures = []
        for (x0a, x1a), (y0a, y1a) in itertools.product(
                zip(x0, x1), zip(y0, y1)):
            futures.append(pool.apply_async(
                mpsegment,
                (x0a-x0p, x1a-x0p, y0a-y0p, y1a-y0p, z0a - z0p, z1a - z0p)))
        for future in tqdm.tqdm(futures):
            c = future.get()
            mask = c[:, -1] >= min_area
            c[:, 0] = c[:, 0] + z0p
            c[:, 1] = c[:, 1] + y0p
            c[:, 2] = c[:, 2] + x0p
            coords.append(c[mask])
    coords = np.vstack(coords)
    return coords[:, :3], coords[:, 3]


def main(args=sys.argv[1:]):
    global x_extent, y_extent, z_extent, stack_files
    args = parse_args(args)
    logging_kwargs = {}
    if args.log_level is not None:
        logging_kwargs["level"] = getattr(logging, args.log_level.upper())
    if args.log_format is not None:
        logging_kwargs["format"] = args.log_format
    if args.log_file is not None:
        logging_kwargs["filename"] = args.log_filename
    logging.basicConfig(**logging_kwargs)
    stack_files = sorted(glob.glob(args.input))
    #
    # Compute the block extents
    #
    z_extent = len(stack_files)
    y_extent, x_extent = tifffile.imread(stack_files[0]).shape
    if args.crop_z is None:
        zmin = 0
        zmax = z_extent
    else:
        zmin, zmax = [int(_) for _ in args.crop_z.split(",")]
    if args.crop_y is None:
        ymin = 0
        ymax = y_extent
    else:
        ymin, ymax = [int(_) for _ in args.crop_y.split(",")]
    if args.crop_x is None:
        xmin = 0
        xmax = x_extent
    else:
        xmin, xmax = [int(_) for _ in args.crop_x.split(",")]

    n_x = (xmax - xmin) // args.block_size_x
    grid_x = np.linspace(xmin, xmax, n_x + 1).astype(int)
    x0 = grid_x[:-1]
    x1 = grid_x[1:]
    n_y = (ymax - ymin) // args.block_size_y
    grid_y = np.linspace(ymin, ymax, n_y + 1).astype(int)
    y0 = grid_y[:-1]
    y1 = grid_y[1:]
    n_z = (zmax - zmin) // args.block_size_z
    grid_z = np.linspace(zmin, zmax, n_z + 1).astype(int)
    z0 = grid_z[:-1]
    z1 = grid_z[1:]
    # Run through a portion of the z-stack per iteration of loop
    # to find the cell centers
    coords = []
    areas = []
    for z0a, z1a in zip(z0, z1):
        c, a = segment_block(x0, x1, y0, y1, z0a, z1a)
        coords.append(c)
        areas.append(a)
    coords = np.vstack(coords)
    areas = np.hstack(areas)
    #
    # Eliminate duplicates if they are too close
    #
    kdt = KDTree(coords)
    too_close = np.array(sorted(kdt.query_pairs(5)))
    reverse = too_close[:, 1] > too_close[:, 0]
    too_close[reverse, 0], too_close[reverse, 1] = \
        too_close[reverse, 1], too_close[reverse, 0]
    to_exclude = np.unique(too_close[:, 1])
    weeded_coords = np.delete(coords, to_exclude, 0)
    weeded_areas = np.delete(areas, to_exclude)
    #
    # This is a fakey JSON writer, just because it may take a long time
    # to write using the json library
    #
    with open(args.output, "w") as fd:
        first_time = True
        for z, y, x in weeded_coords:
            if first_time:
                fd.write("[\n")
                first_time = False
            else:
                fd.write(",\n")
            fd.write("  [%.1f,%.1f,%.1f]" % (x, y, z))
        fd.write("]\n")


if __name__=="__main__":
    main()