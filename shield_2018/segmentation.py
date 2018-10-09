'''segmentation.py - the segmentation method used in the shield paper'''

import argparse
import glob
import itertools
import logging
import multiprocessing
import numpy as np
import tifffile
import tqdm
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_dilation, label
from scipy.spatial import KDTree
from skimage.morphology import watershed
import sys
try:
    from pathom.segmentation.torch_impl import eigvals_of_weingarten
except ImportError:
    from phathom.segmentation.segmentation import eigvals_of_weingarten
from phathom.utils import SharedMemory
from phathom.segmentation.segmentation import adaptive_threshold


class SegmentationParameters:
    """The parameters for a segmentation"""

    def __init__(self):
        self.x_extent = None
        self.y_extent = None
        self.z_extent = None
        self.block_size_x = 1024
        self.block_size_y = 1024
        self.block_size_z = 100
        self.padding = 30
        self.stack_files = None
        #
        # Globals for multiprocessing
        #
        self.io_threads = 4
        self.processing_threads = multiprocessing.cpu_count()
        #
        # Parameters for the segmentation
        #
        """The minimum allowed threshold for the first Otsu"""
        self.t1min = 1.0
        """The maximum allowed threshold for the first Otsu"""
        self.t1max = 5.0
        """The minimum allowed threshold for the second Otsu"""
        self.t2min = 4.0
        """The maximum allowed threshold for the second Otsu"""
        self.t2max = 10.0
        """The minimum area of a segmented object"""
        self.min_area = 20
        """The sigma for the first gaussian of the difference of gaussians"""
        self.dog_low = 3
        """The sigma for the second gaussian of the difference of gaussians"""
        self.dog_high = 10
        """The divisor for the DOG image"""
        self.dog_scaling = 1
        """Whether or not to run adaptive thresholding"""
        self.use_adaptive_threshold = False
        """Smoothing standard deviation applied to adaptive threshold grid"""
        self.adaptive_threshold_sigma = 1.0
        """Adaptive threshold grid size in voxels"""
        self.adaptive_threshold_block_size = 50
        """Base cell centers on seed locations, not the cell centers after watershed"""
        self.use_seed_centers = False
        #
        # These are global shared memory regions for multiprocessing
        #
        """SharedMemory for the image stack"""
        self.stackmem = None


        """SharedMemory for the difference of gaussians"""
        self.dogmem = None
        """Shared memory for the curvature"""
        self.curvmem = None

    @property
    def stack(self):
        with self.stackmem.txn() as m:
            return m

    @property
    def dog(self):
        with self.dogmem.txn() as m:
            return m

    @property
    def curv(self):
        with self.curvmem.txn() as m:
            return m


defaults = SegmentationParameters()


def parse_args(args=sys.argv[1:]):
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
                        default=defaults.block_size_x,
                        help="The size of a processing block in the X direction")
    parser.add_argument("--block-size-y",
                        type=int,
                        default=defaults.block_size_y,
                        help="The size of a processing block in the Y direction")
    parser.add_argument("--block-size-z",
                        type=int,
                        default=defaults.block_size_z,
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
                        default=defaults.padding,
                        help="The amt of padding to add to a block")
    parser.add_argument("--io-threads",
                        type=int,
                        default=defaults.io_threads,
                        help="# of threads to use when reading image files")
    parser.add_argument("--processing-threads",
                        type=int,
                        default=defaults.processing_threads,
                        help="# of threads to use during computation")
    parser.add_argument("--t1min",
                        type=float,
                        default=defaults.t1min,
                        help="The minimum allowed threshold for the low Otsu")
    parser.add_argument("--t1max",
                        type=float,
                        default=defaults.t1max,
                        help="The maximum allowed threshold for the low Otsu")
    parser.add_argument("--t2min",
                        type=float,
                        default=defaults.t2min,
                        help="The minimum allowed threshold for the high Otsu")
    parser.add_argument("--t2max",
                        type=float,
                        default=defaults.t2max,
                        help="The maximum allowed threshold for the hig Otsu")
    parser.add_argument("--dog-low",
                        type=float,
                        default=defaults.dog_low,
                        help="The sigma for the foreground gaussian for the "
                        "difference of gaussians")
    parser.add_argument("--dog-high",
                        type=float,
                        default=defaults.dog_high,
                        help="The sigma for the background gaussian for the "
                        "difference of gaussians")
    parser.add_argument("--dog-scaling",
                        type=float,
                        default=defaults.dog_scaling,
                        help="A divisor for the DOG image to scale its range")
    parser.add_argument("--adaptive-threshold",
                        action="store_true",
                        help="Use a gridded and smoothed adaptive threshold "
                        "within the blocks.")
    parser.add_argument("--adaptive-threshold-sigma",
                        type=float,
                        default=defaults.adaptive_threshold_sigma,
                        help="The smoothing sigma for the adaptive threshold "
                        "at the block grid scale.")
    parser.add_argument("--adaptive-threshold-block_size",
                        type=int,
                        default=defaults.adaptive_threshold_block_size,
                        help="The size of each adaptive threshold block cell "
                        "in voxels")
    parser.add_argument("--seeds-only",
                        action="store_true",
                        help="Base the cell centers only on the location of "
                        "the seed points, foregoing the watershed.")
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


def get_padded_coords(params: SegmentationParameters, x0, x1, y0, y1, z0, z1):
    """Return the block size to use after padding

    :param params: The segmentation parameters
    :param x0: the minimum x coordinate of the region to analyze
    :param x1: the maximum x coordinate of the region to analyze
    :param y0: the minimum y coordinate of the region to analyze
    :param y1: the maximum y coordinate of the region to analyze
    :param z0: the minimum z coordinate of the region to analyze
    :param z1: the maximum z coordinate of the region to analyze
    :return: a tuple of xmin, xmax, ymin, ymax, zmin and zmax for the padded
    region
    """
    z0a = max(z0 - params.padding, 0)
    z1a = min(z1 + params.padding, params.z_extent)
    y0a = max(y0 - params.padding, 0)
    y1a = min(y1 + params.padding, params.y_extent)
    x0a = max(x0 - params.padding, 0)
    x1a = min(x1 + params.padding, params.x_extent)
    return x0a, x1a, y0a, y1a, z0a, z1a


def read_plane(params: SegmentationParameters, z, off=0):
    """Read a plane of memory into the image stack memory

    :param z: the Z coordinate of the plane
    :param off: the offset of the first plane in the stackmem
    """
    with params.stackmem.txn() as m:
        m[z-off] = tifffile.imread(params.stack_files[z])


def compute_dog(params: SegmentationParameters, x0, x1, y0, y1, xoff, yoff):
    """Compute the difference of gaussians for the given region

    Using the image in stackmem, compute the difference of gaussians for a chunk

    :param params: the state variables of the segmentation are in here
    :param x0: the minimum x of the chunk to do
    :param x1: the maximum x of the chunk to do
    :param y0: the minimum y of the chunk to do
    :param y1: the maximum y of the chunk to do
    :param xoff: the x offset into the DOG memory
    :param yoff: the y offset into the DOG memory
    """
    x0a, x1a, y0a, y1a, z0a, z1a = get_padded_coords(
        params, x0, x1, y0, y1, 0, params.dogmem.shape[0])
    with params.stackmem.txn() as stack:
        img = stack[:, y0a:y1a, x0a:x1a].astype(np.float32)
    dog = (gaussian_filter(img, params.dog_low) -
           gaussian_filter(img, params.dog_high)) / params.dog_scaling
    with params.dogmem.txn() as m:
        m[:, y0-yoff:y1-yoff, x0-xoff:x1-xoff] = \
            dog[:, y0-y0a:y1-y0a, x0-x0a:x1-x0a]


def segment(params: SegmentationParameters, x0, x1, y0, y1, z0, z1):
    """Segment a block

    :param params: the parameterization of the segmentation
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
    x0a, x1a, y0a, y1a, z0a, z1a = get_padded_coords(
        params, x0, x1, y0, y1, z0, z1)
    curvv = params.curv[z0a:z1a, y0a:y1a, x0a:x1a]
    curvv[np.isnan(curvv)] = 0
    if params.use_adaptive_threshold:
        t1 = t2 = adaptive_threshold(
            curvv,
            low_threshold=params.t1min,
            high_threshold=params.t1max,
            sigma=params.adaptive_threshold_sigma,
            blocksize=params.adaptive_threshold_block_size)
    else:
        curvvt1 = curvv[(curvv > 0) & (curvv < params.t2max * 2)]
        if len(curvvt1) == 0:
            return np.zeros((0, 4))
        try:
            t1 = min(params.t1max, max(params.t1min, threshold_otsu(curvvt1)))
        except ValueError:
            t1 = params.t1min
        curvvt2 = curvv[(curvv > t1) & (curvv < params.t2max * 2)]
        if len(curvvt2) == 0:
            t2 = t1
        else:
            try:
                t2 = min(
                    params.t2max, max(params.t2min, threshold_otsu(curvvt2)))
            except ValueError:
                t2 = params.t2min
    tseed = (t1 + t2) / 2
    footprint = np.sum(np.square(np.mgrid[-5:6, -5:6, -5:6]), 0) <= 25
    seed_mask = (curvv > tseed) & (grey_dilation(curvv, footprint=footprint) == curvv)
    seeds, count = label(seed_mask)
    if count == 0:
        return np.zeros((0, 4))
    if params.use_seed_centers:
        seg = seeds
    else:
        imask = curvv > t1
        seg = watershed(-curvv, seeds, mask=imask)
    zs, ys, xs = np.where(seg > 0)
    a = np.bincount(seg[zs, ys, xs])[1:]
    zc, yc, xc = [np.bincount(seg[zs, ys, xs], s)[1:] / a + off
                  for s, off in ((zs, z0a), (ys, y0a), (xs, x0a))]
    mask = (zc >= z0) & (zc < z1) & (yc >= y0) & (yc < y1) & (xc >= x0) & (xc < x1)
    return np.column_stack((zc[mask], yc[mask], xc[mask], a[mask]))


def segment_block(params: SegmentationParameters, x0, x1, y0, y1, z0a, z1a):
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
    coords = []
    _, _, _, _, z0p, z1p = get_padded_coords(params,
        0, params.x_extent, 0, params.y_extent, z0a, z1a)
    logging.info("Processing %d to %d" % (z0a, z1a))
    params.stackmem = SharedMemory(
        (z1p - z0p, params.y_extent, params.x_extent), np.uint16)
    with multiprocessing.Pool(params.io_threads) as pool:
        futures = []
        logging.info("Enqueueing reads")
        for z in range(z0p, z1p):
            futures.append(pool.apply_async(read_plane, (params, z, z0p)))
        logging.info("Processing %d reads" % len(futures))
        for future in tqdm.tqdm(futures):
            future.get()
    x0p, x1p, y0p, y1p, _, _ = get_padded_coords(params, x0[0], x1[-1],
                                                 y0[0], y1[-1],
                                                 0, z1p - z0p)
    params.dogmem = SharedMemory((z1p - z0p, y1p-y0p, x1p-x0p), np.float32)
    futures = []
    with multiprocessing.Pool(params.processing_threads) as pool:
        for (x0a, x1a), (y0a, y1a) in itertools.product(
                zip(x0, x1), zip(y0, y1)):
            futures.append(pool.apply_async(
                compute_dog, (params, x0a, x1a, y0a, y1a, x0p, y0p)))
        for future in tqdm.tqdm(futures):
            future.get()
    params.stackmem = None
    params.curvmem = SharedMemory((z1p - z0p, y1p - y0p, x1p - x0p),
                                  np.float32)
    with params.dogmem.txn() as dog:
        e = eigvals_of_weingarten(dog[:, :y1p-y0p, :x1p-x0p])
        params.curv[:] = np.all(e < 0, -1) * e[..., 0] * e[..., 1]
    params.dogmem = None
    with multiprocessing.Pool(params.processing_threads) as pool:
        futures = []
        for (x0a, x1a), (y0a, y1a) in itertools.product(
                zip(x0, x1), zip(y0, y1)):
            futures.append(pool.apply_async(
                segment,
                (params,
                 x0a-x0p, x1a-x0p,
                 y0a-y0p, y1a-y0p,
                 z0a - z0p, z1a - z0p)))
        for future in tqdm.tqdm(futures):
            c = future.get()
            c[:, 0] = c[:, 0] + z0p
            c[:, 1] = c[:, 1] + y0p
            c[:, 2] = c[:, 2] + x0p
            if params.use_seed_centers:
                coords.append(c)
            else:
                mask = c[:, -1] >= params.min_area
                coords.append(c[mask])
    coords = np.vstack(coords)
    return coords[:, :3], coords[:, 3]


def main(args=sys.argv[1:]):
    args = parse_args(args)
    do_segmentation(
        args.input,
        args.output,
        block_size_x = args.block_size_x,
        block_size_y = args.block_size_y,
        block_size_z = args.block_size_z,
        crop_x = args.crop_x,
        crop_y = args.crop_y,
        crop_z =args.crop_z,
        padding = args.padding,
        io_threads = args.io_threads,
        processing_threads = args.processing_threads,
        t1min = args.t1min,
        t1max = args.t1max,
        t2min = args.t2min,
        t2max = args.t2max,
        dog_low = args.dog_low,
        dog_high = args.dog_high,
        dog_scaling=args.dog_scaling,
        use_adaptive_threshold = args.adaptive_threshold,
        adaptive_threshold_sigma = args.adaptive_threshold_sigma,
        adaptive_threshold_block_size = args.adaptive_threshold_block_size,
        log_level = args.log_level,
        log_filename = args.log_file,
        log_format = args.log_format
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
        io_threads=defaults.io_threads,
        processing_threads=defaults.processing_threads,
        t1min=defaults.t1min,
        t1max=defaults.t1max,
        t2min=defaults.t2min,
        t2max=defaults.t2max,
        dog_low=defaults.dog_low,
        dog_high=defaults.dog_high,
        dog_scaling=defaults.dog_scaling,
        use_adaptive_threshold=defaults.use_adaptive_threshold,
        adaptive_threshold_sigma=defaults.adaptive_threshold_sigma,
        adaptive_threshold_block_size=defaults.adaptive_threshold_block_size,
        use_seed_centers=defaults.use_seed_centers,
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
    :param use_adaptive_threshold: Use a gridded and smoothed adaptive threshold
                    instead of a single threshold per processing block
    :param adaptive_threshold_sigma: The smoothing standard deviation applied
    to the adaptive threshold grid, if use_adaptive_threshold is used.
    :param adaptive_threshold_block_size: The size of each grid block in voxels
    :param use_seed_centers: forgo the watershed and use the seeds as cell
                             centers
    :param log_level: The log level for the Python logger: one of "
                      "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    :param log_filename: File to log to. Default is console.
    :param log_format: Format for log messages. See "
                        "https://docs.python.org/3/howto/logging.html"
                        "#changing-the-format-of-displayed-messages for help
    """
    params = SegmentationParameters()
    for param_name in dir(params):
        if param_name in locals():
            setattr(params, param_name, locals()[param_name])

    logging_kwargs = {}
    if log_level is not None:
        logging_kwargs["level"] = getattr(logging, log_level.upper())
    if log_format is not None:
        logging_kwargs["format"] = log_format
    if log_filename is not None:
        logging_kwargs["filename"] = log_filename
    logging.basicConfig(**logging_kwargs)
    params.stack_files = sorted(glob.glob(input))
    #
    # Compute the block extents
    #
    params.z_extent = len(params.stack_files)
    try:
        params.y_extent, params.x_extent = \
            tifffile.imread(params.stack_files[0]).shape
    except IndexError:
        sys.stderr.write("Could not find stack files at %s" % input)
        exit(1)
    if crop_z is None:
        zmin = 0
        zmax = params.z_extent
    else:
        zmin, zmax = [int(_) for _ in crop_z.split(",")]
    if crop_y is None:
        ymin = 0
        ymax = params.y_extent
    else:
        ymin, ymax = [int(_) for _ in crop_y.split(",")]
    if crop_x is None:
        xmin = 0
        xmax = params.x_extent
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
        c, a = segment_block(params, x0, x1, y0, y1, z0a, z1a)
        coords.append(c)
        areas.append(a)
    coords = np.vstack(coords)
    areas = np.hstack(areas)
    #
    # Eliminate duplicates if they are too close
    #
    if len(coords) > 0:
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
    else:
        weeded_coords = coords
        weeded_areas = areas
    #
    # This is a fakey JSON writer, just because it may take a long time
    # to write using the json library
    #
    with open(output, "w") as fd:
        first_time = True
        fd.write("[\n")
        for z, y, x in weeded_coords:
            if first_time:
                first_time = False
            else:
                fd.write(",\n")
            fd.write("  [%.1f,%.1f,%.1f]" % (x, y, z))
        fd.write("]\n")


if __name__=="__main__":
    main()