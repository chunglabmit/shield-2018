# Shield-2018

This is a collection of the software used in the Chung Lab's 2018 Shield
paper. It is packaged as a docker container (see https://www.docker.com/)
so that it can be easily and accurately deployed in labs with a minimum of
IT administration support. The dockerfiles can also be used as recipes to
build the components.

The software handles the following tasks:

* Stitching and destriping volumes acquired from a SPIM microscope or
similar. Destriping is performed using PyStripe 
(https://github.com/chunglabmit/pystripe). The stitching alignment is
performed using Terastitcher (https://github.com/abria/TeraStitcher) and
the aligned volume is written to disk using TSV 
(https://github.com/chunglabmit/tsv).

* Alignment to the Allen Brain Atlas. This is done using Nuggt
(https://github.com/chunglabmit/nuggt). The alignment pipeline that was
used is outlined here: https://github.com/chunglabmit/nuggt#alignment-pipeline

* Segmentation. The shield-2018 package includes a command, 
**shield-2018-segmentation**, that was used to find putative cell centers in
the paper. These were then proofread using **nuggt**. The method robustly
finds roughly spherical light objects against a dark background but finds
false positives in areas of fiber. It is suggested that users supplant this
with either manual editing or a machine-learning method that classifies
true and false positive detections. The command parameters are detailed
below.

* Assignment of detections to the atlas. This is done using the 
[count-points-in-region](https://github.com/chunglabmit/nuggt#count-points-in-region)
command.

## Installation

The preferred method of installation of **shield-2018** is as a docker
container. If you have Docker installed, this is available as

```commandline
> docker pull chunglabmit/shield-2018
```

A typical invocation might be
```commandline
docker run -v /path-to-stack:/stack \
           -v /path-to-output:/output \
           chunglabmit/shield-2018 \
           shield-2018-segmentation \
           --input /stack/img_*.tiff \
           --output /output/result.json
```

Commands from the Terastitcher and Nuggt packages can also be run from
this docker. See [the Nuggt documentation](https://github.com/LeeKamentsky/nuggt#docker)
for more help.

## The shield-2018-segment command

The **shield-2018-segment** command finds cell centers in a stack
of images. The output is a .json file containing a list of lists with
each element of the inner list being the Z, Y and X coordinate of one of
the cell centers. The format of the command is:

```commandline
shield-2018-segment \
    --input <input-stack-expression> \
    --output <output-filename> \
    [--block-size-x <block-size-x>] \
    [--block-size-y <block-size-y>] \
    [--block-size-z <block-size-z>] \
    [--crop-x <crop-x>] \
    [--crop-y <crop-y>] \
    [--crop-z <crop-z>] \
    [--padding <padding>] \
    [--io-threads <io-threads>] \
    [--processing-threads <processing-threads>] \
    [--t1min <t1min>] \
    [--t1max <t1max>] \
    [--t2min <t2min>] \
    [--t2max <t2max>] \
    [--dog-low <dog-low>] \
    [--dog-high <dog-high>] \
    [--log-level <log-level>] \
    [--log-file <log-file>] \
    [--log-format <log-format>]
```
where
* **input-stack-expression** is a glob expression for listing the .tif
files in the input stack, e.g. "/path-to/img_*.tif". The files are read
in alphabetical order. Typically, the names are zero-padded, e.g.
"img_0001.tif", so that they will be read in the correct order.
* **output-filename** the name of the .json file containing the cell
centers.
* **block-size-x**, **block-size-y**, **block-size-z** The processing is
done by breaking the volume into blocks in order to process teravoxel
volumes on machines with less than terabyte memory sizes. These parameters
specify the size of a block. Segmentation requires enough memory to hold
(**block-size-z** + 2 * (**padding**)) * 32 bytes of memory for each pixel in
one image plane and an additional **block-size-x** * **block-size-y** *
**block-size-z** * **processing-threads** * 32 bytes to handle
parallel processing.
* **crop-x**, **crop-y**, **crop-z** Processing can be done on a portion of the
stack (or processing can be broken up into manageable chunks and combined)
by cropping the stack to the dimensions given by these parameters.
Each one is specified as a minimum and maximum coordinate, e.g.
`--crop-x 1000,3000` processes the portion of the stack from x=1000 to
x=3000.
* **padding** is the amount of padding to add to the X, Y and Z boundaries
of a block. This should be roughly 3x the background sigma specified
by the `--dog-high` switch. The default is 30.
* **io-threads** is the number of threads used when reading the image
stack. The default is 4.
* **processing-threads** is the number of threads used during parallel
processing. The default is the number of cores on the computer being used.
* **t1-min**, **t1-max**, **t2-min**, **t2-max** These are threshold
limits for the seed-finding and segmentation thresholding step. T1 is
the lower threshold used for segmentation and (T1+T2) / 2 is the
threshold used for the seed finding step. These numbers can be adjusted
if the method is finding false positive or false negative cell centers.
* **dog-low**, **dog-high** These are the sigmas for the difference of
gaussians. **dog-low** is designed to pick the scale of the cells and
should be roughly 1/4 of the cell width in voxels. **dog-high** is
designed to pick the scale of the background (which is subtracted from
the foreground signal). The defaults are 3 and 10.
* **log-level** is the minimum log message level. Possible values are
"DEBUG", "INFO", "WARNING", "ERROR" or "CRITICAL". The default is
 "WARNING".
* **log-file** is the path to the file in which to record the log
messages. The default is to display log messages on the console.
* **log-format** is the display format for log messages. See
https://docs.python.org/3/howto/logging.html#changing-the-format-of-displayed-messages
for help.