************
How to Start
************
After running installation script in OSX or Windows there should be a doulbe-click executable named `TierpsyTracker` in the Desktop. If the executable is missing you can run ``installation/installation_script.sh--link_desktop`` in OSX or ``installation/installation_script.bat --link_desktop`` in Windows to re-create the executable.

The alternative is to open a terminal, move to the Tierpsy Tracker main directory and type ``python3 cmd_scripts/TierpsyTrackerConsole.py``.


*******
Options
*******
.. image:: https://cloud.githubusercontent.com/assets/8364368/26286115/0b0c7376-3e55-11e7-918c-cc0319b90496.png
   :align: center
   


Set Parameters
==============
- If the worm density is high, the worm occupies a large area of the field of view or the raw video is already heavily compressed, the output hdf5 file can be larger than the original video.
 - The hdf5 storage of the masked images is important in our setup: the high resolution and high-througput make even jpg compressed videos too large to be kept for long time storage. However, in the future this step might be done in real time in our system. 
.. image:: https://cloud.githubusercontent.com/assets/8364368/26286358/793a423c-3e5b-11e7-8e8f-f94da9c26ba9.gif
.. image:: https://cloud.githubusercontent.com/assets/8364368/26287848/475e64f8-3e7b-11e7-8a1c-d4d94dbbcf59.gif




Batch Processing Multiple Files
===============================
It is a GUI for command line tool ProcessMultipleFiles.py.

Single-Worm Tracker Viewer
===============================

Multi-Worm Tracker Viewer
===============================
