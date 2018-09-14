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
                        default=100,
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
    args = parse_args(args)
    do_segmentation(
        args.input,
        args.output,
        args.block_size_x,
        args.block_size_y,
        args.block_size_z,
        args.crop_x,
        args.crop_y,
        args.crop_z,
        args.padding,
        args.io_threads,
        args.processing_threads,
        args.t1min,
        args.t1max,
        args.t2min,
        args.t2max,
        args.dog_low,
        args.dog_high,
        args.log_level,
        args.log_file,
        args.log_format
    )

def do_segmentation(
        input,
        output,
        block_size_x=1024,
        block_size_y=1024,
        block_size_z=100,
        crop_x=None,
        crop_y=None,
        crop_z=None,
        padding=30,
        io_threads=io_threads,
        processing_threads=processing_threads,
        t1min=t1min,
        t1max=t1max,
        t2min=t2min,
        t2max=t2max,
        dog_low=dog_low,
        dog_high=dog_high,
        log_level="INFO",
        log_filename=None,
        log_format=None):
    """Run the segmentation pipeline

    :param input: A glob expression for the stack of images to
                  analyze, for instance, "/path-to/img_*.tiff".
    :param output: The path to the output json file of cell centers
    :param block_size_x: The size of a processing block in the X direction
    :param block_size_y: The size of a processing block in the Y direction
    :param block_size_z: The size of a processing block in the Z direction.
    It may be necessary to set this in order to reduce memory consumption.
    :param crop_x: Crop the volume to these x coordinates.
                   Specify as a two-tuple of minimum and maximum coordinates.
    :param crop_y: Crop the volume to these y coordinates.
                   Specify as a two-tuple of minimum and maximum coordinates.
    :param crop_z:  Crop the volume to these z coordinates.
                   Specify as a two-tuple of minimum and maximum coordinates.
    :param padding: The amt of padding to add to a block
    :param io_threads: # of threads to use when reading image files
    :param processing_threads: # of threads to use during computation
    :param t1min: The minimum allowed threshold for the low Otsu
    :param t1max: The maximum allowed threshold for the low Otsu
    :param t2min: The minimum allowed threshold for the high Otsu
    :param t2max: The maximum allowed threshold for the high Otsu
    :param dog_low: The sigma for the foreground gaussian for the
                    difference of gaussians
    :param dog_high: The sigma for the background gaussian for the
                     difference of gaussians
    :param log_level: The log level for the Python logger: one of "
                      "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    :param log_filename: File to log to. Default is console.
    :param log_format: Format for log messages. See "
                        "https://docs.python.org/3/howto/logging.html"
                        "#changing-the-format-of-displayed-messages for help
    """
    global x_extent, y_extent, z_extent, stack_files
    for name in ("io_threads", "processing_threads", "padding",
        "t1min", "t1max", "t2min", "t2max", "dog_low", "dog_high"):
        globals()[name] = locals()[name]

    logging_kwargs = {}
    if log_level is not None:
        logging_kwargs["level"] = getattr(logging, log_level.upper())
    if log_format is not None:
        logging_kwargs["format"] = log_format
    if log_filename is not None:
        logging_kwargs["filename"] = log_filename
    logging.basicConfig(**logging_kwargs)
    stack_files = sorted(glob.glob(input))
    #
    # Compute the block extents
    #
    z_extent = len(stack_files)
    try:
        y_extent, x_extent = tifffile.imread(stack_files[0]).shape
    except IndexError:
        sys.stderr.write("Could not find stack files at %s" % input)
        exit(1)
    if crop_z is None:
        zmin = 0
        zmax = z_extent
    else:
        zmin, zmax = [int(_) for _ in crop_z.split(",")]
    if crop_y is None:
        ymin = 0
        ymax = y_extent
    else:
        ymin, ymax = [int(_) for _ in crop_y.split(",")]
    if crop_x is None:
        xmin = 0
        xmax = x_extent
    else:
        xmin, xmax = [int(_) for _ in crop_x.split(",")]

    n_x = max(1, (xmax - xmin) // block_size_x)
    grid_x = np.linspace(xmin, xmax, n_x + 1).astype(int)
    x0 = grid_x[:-1]
    x1 = grid_x[1:]
    n_y = max(1, (ymax - ymin) // block_size_y)
    grid_y = np.linspace(ymin, ymax, n_y + 1).astype(int)
    y0 = grid_y[:-1]
    y1 = grid_y[1:]
    n_z = max(1, (zmax - zmin) // block_size_z)
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
    if len(too_close) > 0:
        reverse = too_close[:, 1] > too_close[:, 0]
        too_close[reverse, 0], too_close[reverse, 1] = \
            too_close[reverse, 1], too_close[reverse, 0]
        to_exclude = np.unique(too_close[:, 1])
        weeded_coords = np.delete(coords, to_exclude, 0)
        weeded_areas = np.delete(areas, to_exclude)
    else:
        weeded_coords = coords
        weeded_areas = areas
    #
    # This is a fakey JSON writer, just because it may take a long time
    # to write using the json library
    #
    with open(output, "w") as fd:
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