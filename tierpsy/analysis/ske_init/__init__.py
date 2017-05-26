from .processTrajectoryData import processTrajectoryData

def args_(fn, param):
    #arguments used by AnalysisPoints.py
    return {
        'func': processTrajectoryData,
        'argkws': {**{'skeletons_file': fn['skeletons'], 
                    'masked_image_file':fn['masked_image'],
                    'trajectories_file': fn['skeletons']},
                    **param.init_skel_param},

        'input_files' : [fn['masked_image'], fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['TRAJ_JOIN'],
    }