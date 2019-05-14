# This runs the protocol described in
# Swaney, "Scalable image processing techniques for
# quantitative analysis of volumetric biological images
# from light-sheet microscopy"
#
# You can run it in the docker. You may want to do
# docker run --shm-size 1g -v <protocol-dir>:/protocol /bin/bash tests/run_protocol.sh
#
apt install unzip
mkdir -p /protocol
cd /protocol
wget https://leviathan-chunglab.mit.edu/nature-protocols-2019/downsampled_data.zip
unzip downsampled_data.zip
pystripe -i /protocol/downsampled_data/raw_data/Ex_0_Em_0 \
         -o /protocol/downsampled_data/destriped_data/Ex_0_Em_0 \
         -s1 32 -s2 32 -w db10 -c 3 -x 10 \
         --flat /protocol/downsampled_data/raw_data/flat_downsampled.tif \
         --dark 100
pystripe -i /protocol/downsampled_data/raw_data/Ex_1_Em_1 \
         -o /protocol/downsampled_data/destriped_data/Ex_1_Em_1 \
         -s1 32 -s2 32 -w db10 -c 3 -x 10 \
         --flat /protocol/downsampled_data/raw_data/flat_downsampled.tif \
         --dark 100
pystripe -i /protocol/downsampled_data/raw_data/Ex_2_Em_2 \
         -o /protocol/downsampled_data/destriped_data/Ex_2_Em_2 \
         -s1 32 -s2 32 -w db10 -c 3 -x 10 \
         --flat /protocol/downsampled_data/raw_data/flat_downsampled.tif \
         --dark 100
terastitcher -1 --volin=/protocol/downsampled_data/destriped_data/Ex_1_Em_1 \
  --ref1=H --ref2=V --ref3=D --vxl1=14.4 --vxl2=14.4 --vxl3=16 \
  --projout=/protocol/downsampled_data/destriped_data/Ex_1_Em_1/xml_import.xml \
  --sparse_data
terastitcher -2 \
  --projin=/protocol/downsampled_data/destriped_data/Ex_1_Em_1/xml_import.xml \
  --projout=/protocol/downsampled_data/destriped_data/Ex_1_Em_1/xml_displacement.xml
terastitcher -3 \
  --projin=/protocol/downsampled_data/destriped_data/Ex_1_Em_1/xml_displacement.xml \
  --projout=/protocol/downsampled_data/destriped_data/Ex_1_Em_1/xml_displproj.xml
tsv-convert-2D-tif \
  --xml-path /protocol/downsampled_data/destriped_data/Ex_1_Em_1/xml_displproj.xml \
  --output-pattern /protocol/downsampled_data/stitched_data/Ex_1_Em_1_master/"{z:04d}.tiff" \
  --compression 4 --ignore-z-offsets
tsv-convert-2D-tif \
  --xml-path /protocol/downsampled_data/destriped_data/Ex_1_Em_1/xml_displproj.xml \
  --output-pattern /protocol/downsampled_data/stitched_data/Ex_0_Em_0/"{z:04d}.tiff" \
  --compression 4 --ignore-z-offsets \
  --input /protocol/downsampled_data/destriped_data/Ex_0_Em_0
tsv-convert-2D-tif \
  --xml-path /protocol/downsampled_data/destriped_data/Ex_1_Em_1/xml_displproj.xml \
  --output-pattern /protocol/downsampled_data/stitched_data/Ex_2_Em_2/"{z:04d}.tiff" \
  --compression 4 --ignore-z-offsets \
  --input /protocol/downsampled_data/destriped_data/Ex_2_Em_2

rescale-image-for-alignment \
  --input "/protocol/downsampled_data/stitched_data/Ex_0_Em_0/*.tiff" \
  --atlas-file /allen-mouse-brain-atlas/autofluorescence_25_half_sagittal.tif \
  --output /protocol/downsampled_data/downsampled_flip-x_flip-z_clip-y-0-1225.tiff \
  --flip-x --flip-z --clip-y 0,1225
sitk-align \
  --moving-file /protocol/downsampled_data/downsampled_flip-x_flip-z_clip-y-0-1225.tiff \
  --fixed-file /allen-mouse-brain-atlas/autofluorescence_25_half_sagittal.tif \
  --fixed-point-file /allen-mouse-brain-atlas/coords_25_half_sagittal.json \
  --xyz --alignment-point-file /protocol/downsampled_data/alignment.json

#
# Here, someone might run nuggt-align
#
rescale-alignment-file \
  --stack "/protocol/downsampled_data/stitched_data/Ex_0_Em_0/*.tiff" \
  --alignment-image /protocol/downsampled_data/downsampled_flip-x_flip-z_clip-y-0-1225.tiff \
  --input /protocol/downsampled_data/alignment.json \
  --output /protocol/downsampled_data/rescaled-alignment.json \
  --flip-x --flip-z --clip-y 0,1225
calculate-intensity-in-regions \
  --input "/protocol/downsampled_data/stitched_data/Ex_1_Em_1_master/*.tiff" \
  --alignment /protocol/downsampled_data/rescaled-alignment.json \
  --reference-segmentation /allen-mouse-brain-atlas/annotation_25_half_sagittal.tif \
  --brain-regions-csv /allen-mouse-brain-atlas/AllBrainRegions.csv \
  --output /protocol/downsampled_data/results-level-3.csv --level 3 \
  --output /protocol/downsampled_data/results-level-4.csv --level 4 \
  --output /protocol/downsampled_data/results-level-5.csv --level 5 \
  --output /protocol/downsampled_data/results-level-6.csv --level 6 \
  --output /protocol/downsampled_data/results-level-7.csv --level 7
