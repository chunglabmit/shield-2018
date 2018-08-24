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