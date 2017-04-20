import tables
import numpy as np


class AttrReader():
    valid_fields = ['/mask', '/trajectories_data', '/features_timeseries']
    
    def __init__(file_name, dflt=1):
        self.file_name = file_name
        self.dflt = dflt
        self.field =  self._find_field()



    def _find_field(self):
        with tables.File(self.file_name, 'r') as fid:
            for field in valid_fields:
                if field in fid:
                    return field
        raise KeyError("Not valid field {} found in {}".format(valid_fields, fname)) 


    def _read_expected_attr(attr_name):
        with tables.File(self.file_name, 'r') as fid:
            node = fid.get_node(self.field)
            if attr_name in node._v_attrs:
                attr = node._v_attrs[attr_name]
            else:
                attr = self.dflt #default in old videos
            return attr

    def get_fps():
        try:
            #try to calculate the frames per second from the timestamp
            with tables.File(self.file_name, 'r') as fid:
                timestamp_time = fid.get_node('/timestamp/time')[:]

                if np.all(np.isnan(timestamp_time)):
                    raise ValueError
                fps = 1 / np.nanmedian(np.diff(timestamp_time))

                if np.isnan(fps) or fps < 1:
                    raise ValueError
                is_user_timestamp = 0

        except (tables.exceptions.NoSuchNodeError, IOError, ValueError):
            #read the user defined timestamp
            fps = self._read_expected_attr('expected_fps', dflt=dflt)
            is_user_timestamp = 1
        
        self._fps, self.is_user_timestamp = fps, is_user_timestamp

        return self._fps, self.is_user_timestamp

    @property
    def fps(self):
        try:
            return self._fps
        except:
            self.get_fps()
            return self._fps

    @property
    def is_user_timestamp(self):
        try:
            return self._is_user_timestamp
        except:
            self.get_fps()
            return self._is_user_timestamp

    @property
    def microns_per_pixel(self):
        try:
            return self._microns_per_pixel
        except:
            try:
                #this for the shaffer's lab single worm case...
                with tables.File(self.file_name, 'r') as fid:
                    microns_per_pixel_scale = fid.get_node('/stage_movement')._v_attrs['microns_per_pixel_scale']
                    if microns_per_pixel_scale.size == 2:
                        assert np.abs(
                            microns_per_pixel_scale[0]) == np.abs(
                            microns_per_pixel_scale[1])
                        microns_per_pixel = np.abs(microns_per_pixel_scale[0])
                
            except (KeyError, tables.exceptions.NoSuchNodeError):
                microns_per_pixel = self._read_expected_attr('microns_per_pixel', dflt=dflt)
            
            self._microns_per_pixel = microns_per_pixel
            return self._microns_per_pixel

     @property
    def is_light_background(self):
        try:


def add_unit_conversions(group_to_save, original_file):
    microns_per_pixel = read_microns_per_pixel(original_file, dflt=np.nan)
    fps, is_user_timestamp = read_fps(original_file, dflt=np.nan)
    
    # save some data used in the calculation as attributes
    group_to_save._v_attrs['microns_per_pixel'] = microns_per_pixel
    group_to_save._v_attrs['is_user_timestamp'] = is_user_timestamp #if the data was calculated from the video timestamps or from a user given value
    group_to_save._v_attrs['fps'] = fps
    
    return fps, microns_per_pixel


