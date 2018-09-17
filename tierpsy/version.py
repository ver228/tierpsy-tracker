# # -*- coding: utf-8 -*-
__version__ = '1.5.1-beta2'

try:
    import os
    import subprocess

    cwd = os.path.dirname(os.path.abspath(__file__)) 
    command = ['git', 'rev-parse', 'HEAD']    
    sha = subprocess.check_output(command, cwd=cwd, stderr = subprocess.DEVNULL).decode('ascii').strip()
    __version__ += '+' + sha[:7]

except Exception:
    pass

'''
1.5.1-beta
- Create a conda package. This will be the new way to distribute the package.
- Merge part of the code in tierpsy-features and open-worm-analysis-toolbox to remove them as dependencies.
- Merge test suite into the main package.

1.5.0
- Bug corrections.

1.5.0-beta
- Complete the integration with tierpsy features formalizing two different feature paths.
- Reorganize the types of analysis, and deal with deprecated values.
- Add plot feature option in the tracker viewer to visualize individual worm features.
- Make the tracker viewer compatible with the _features.hdf5 files and deprecate the WT2 viewer
- Add app to collect the feature summary of different videos.



1.5.0-alpha
- Add tierpsy features as FEAT_INIT, FEAT_TIERPSY.
- Reorganize and improve GUIs, particularly "Set Parameters".
- Fix background subtraction to deal with light background.
- Add analysis point ('WORM_SINGLE') to tell the algorithm that it is expected only one worm in the video.
- Make the calculation of FPS from timestamp more tolerant to missed/repeated frames.
- Change head/tail identification to deal with worms with very low to none global displacement.

1.4.0
- Schafer's lab tracker ready for release:
	* Remove CNT_ORIENT as a separated checkpoint and add it rather as a pre-step using a decorator.
	* Stage aligment failures throw errors instead of continue silently and setting an error flag.
- Bug fixes

1.4.0b0
- Remove MATLAB dependency.
- Uniformly the naming event features (coil/coils omega_turn/omega_turns forward_motion/forward ...)
- Add food features and food contour analysis (experimental)
- Improvements to the GUI
- Bug fixes

1.3
- Major internal organization.
- Documentation
- First oficial release.

1.2.1
- Major changes in internal organization of TRAJ_CREATE TRAJ_JOIN
- _trajectories.hdf5 is deprecated. The results of this file are going to be saved in _skeletons.hdf5
- GUI Multi-worm tracker add the option of show trajectories.

1.2.0
- Major refactoring
- Add capability of identifying worms using a pre-trained neural network (not activated by default).
- Separated the creation of the control table in the skeletons file (trajectories_data) from the actually 
skeletons calculation. The point SKEL_INIT now preceds SKEL_CREATION.

1.1.1
- Cumulative changes and bug corrections.

1.1.0 
- Fish tracking (experimental) and fluorescence tracking.
- major changes in the internal paralization, should make easier to add or remove steps on the tracking.
- correct several bugs.
'''