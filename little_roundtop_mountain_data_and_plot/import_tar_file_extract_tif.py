# importing the "tarfile" module
import tarfile
  
# open file
file = tarfile.open('little_roundtop_mountain_data_and_plot\little_roundtop_raster.tar.gz')
  
# extracting file
file.extractall('./little_roundtop_mountain_data_and_plot')
  
file.close()