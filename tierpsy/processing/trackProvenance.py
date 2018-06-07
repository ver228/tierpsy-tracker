import os
import json
import tables
import importlib

def getPackagesVersion():
    pkgs_versions = {}
    for pkg in ['tierpsy', 'open_worm_analysis_toolbox', 'tierpsy_features']:
        try:
            mod = importlib(pkg)
            ver = mod.__version__
        except:
            ver = ''
        pkgs_versions[pkg] = ver


    return pkgs_versions


def execThisPoint(
        thisPoint,
        func,
        argkws,
        provenance_file,
        pkgs_versions,
        cmd_original):
    # execute the function
    func(**argkws)

    # make sure the file was created correctly
    assert os.path.exists(provenance_file)

    # this are the variables that are going to be stored
    variables2save = {
        'func_name': func.__name__,
        'func_arguments': json.dumps(argkws),
        'pkgs_versions': pkgs_versions,
        'cmd_original': cmd_original}
    variables2save = bytes(json.dumps(variables2save), 'utf-8')

    # store the variables in the correct node
    with tables.File(provenance_file, 'r+') as fid:
        main_node = 'provenance_tracking'
        if not '/' + main_node in fid:
            fid.create_group('/', main_node)
        if '/'.join(['', main_node, thisPoint]) in fid:
            fid.remove_node('/' + main_node, thisPoint)

        fid.create_array('/' + main_node, thisPoint, obj=variables2save)
        fid.flush()
