import glob
import h5py
import numpy as np

rootDir = "/Volumes/behavgenom$/GeckoVideo/MaskedVideos/"


file_list = glob.glob(rootDir + "*/*.hdf5")

for name in file_list:
	print(name)
	with h5py.File(name, 'r+') as fid:
		mask_dataset = fid['/mask']
		full_dataset = fid['/full_data']

		#labels to make the group compatible with the standard image definition in hdf5
		mask_dataset.attrs["CLASS"] = np.string_("IMAGE")
		mask_dataset.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_GRAYSCALE")
		mask_dataset.attrs["IMAGE_WHITE_IS_ZERO"] = np.array(0, dtype="uint8")
		mask_dataset.attrs["DISPLAY_ORIGIN"] = np.string_("UL") # not rotated
		mask_dataset.attrs["IMAGE_VERSION"] = np.string_("1.2")

		#labels to make the group compatible with the standard image definition in hdf5
		full_dataset.attrs["CLASS"] = np.string_("IMAGE")
		full_dataset.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_GRAYSCALE")
		full_dataset.attrs["IMAGE_WHITE_IS_ZERO"] = np.array(0, dtype="uint8")
		full_dataset.attrs["DISPLAY_ORIGIN"] = np.string_("UL") # not rotated
		full_dataset.attrs["IMAGE_VERSION"] = np.string_("1.2")