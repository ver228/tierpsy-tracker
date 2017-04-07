
import tables
import numpy as np
from .read_attrs import read_fps

class CorrectParams():
    def __init__(self, main_arg):
        self.main_arg=  main_arg

    def process(self, params, convertions):
        for key in convertions:
            if key in params:
                param = params[key]
                if param is None or param <=0:
                    params[key] = convertions[key]()
        return params


    @property
    def fps(self):
        try:
            return self._fps
        except:
            if isinstance(self.main_arg, str):
                self._fsp, _ = read_fps(self.main_arg)
            else:
                print(type(self.main_arg))
                self._fsp = self.main_arg
            return self._fsp

    @property
    def resampling_N(self):
        try:
            return self._resampling_N
        except:
            assert isinstance(self.main_arg, str)
            with tables.File(self.main_arg, 'r') as fid:
                self._resampling_N = fid.get_node('/skeleton').shape[1]
                return self._resampling_N



def compress_defaults(expected_fps, **params):
    obj = CorrectParams(expected_fps)

    convertions = dict(
        buffer_size = lambda: int(round(obj.fps)),
        save_full_interval = lambda: int(200 * obj.fps)
    )
    
    return obj.process(params, convertions)

def traj_create_defaults(fname, buffer_size):
    output = compress_defaults(fname, buffer_size=buffer_size)
    return output['buffer_size']

def ske_init_defaults(fname, **params):
    obj = CorrectParams(fname)

    convertions = dict(
        displacement_smooth_win = lambda: int(round(obj.fps+1)),
        threshold_smooth_win = lambda: int(round((obj.fps*20)+1))
    )
    return obj.process(params, convertions)
    
def head_tail_defaults(fname, **params):
    obj = CorrectParams(fname)

    convertions = dict(
        max_gap_allowed = lambda: max(1, int(obj.fps//2)),
        window_std = lambda: int(round(obj.fps)),
        min_block_size = lambda: int(round(10 * obj.fps)),
        segment4angle = lambda: int(round(obj.resampling_N / 10))
    )
    return obj.process(params, convertions)

def head_tail_int_defaults(fname, **params):
    
    obj = CorrectParams(fname)

    convertions = dict(
        smooth_W = lambda: max(1, int(round(obj.fps / 5))),
        gap_size = lambda: max(1, int(obj.fps//4)),
        min_block_size = lambda: int(round(2*obj.fps/5)),
        local_avg_win = lambda: int(round(10 * obj.fps))
    )
    return obj.process(params, convertions)

def min_num_skel_defaults(fname, **params):
    obj = CorrectParams(fname)
    convertions = dict(
        min_num_skel = lambda: int(round(4 * obj.fps))
        )
    output = obj.process(params, convertions)
    if len(output) == 1:
        return output['min_num_skel']
    else:
        return output


if __name__ == '__main__':
    fname = '/Volumes/Samsung USB/unc-4/Results/L4_unc4(e2323)_Ch1_31032017_140732_skeletons.hdf5'
    output = compress_defaults(50, buffer_size=-1, save_full_interval=None)
    print(output)

