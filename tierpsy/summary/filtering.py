#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:24:21 2020

@author: em812
"""
import pandas as pd

def _traj_lenght(timeseries_data, min_traj_length):

    traj_lengths = timeseries_data['timestamp'].groupby(
        by = timeseries_data['worm_index']).nunique()
    keep = traj_lengths[traj_lengths > min_traj_length].index.to_list()

    return timeseries_data.loc[timeseries_data['worm_index'].isin(keep), :]

def _distance_traveled(timeseries_data, min_distance_traveled):
    grouped = timeseries_data.groupby(by='worm_index')
    dist = []
    for body_part in ['body', 'tail', 'midbody', 'head']:
        coords = ['_'.join([x, body_part])  for x in ['coord_x', 'coord_y']]
        dist.append(grouped[coords].apply(
            lambda x: x.diff().pow(2).sum(1).pow(0.5).sum()))
    dist = pd.concat(dist, axis=1)
    keep = dist[(dist > min_distance_traveled).any(axis=1)].index.to_list()
    return timeseries_data.loc[timeseries_data['worm_index'].isin(keep), :]

def _by_timeseries_values(
        timeseries_data, name, min_threshold=None, max_threshold=None):

    if min_threshold is None and max_threshold is None:
        return timeseries_data

    mean_values = timeseries_data[name].groupby(
        by = timeseries_data['worm_index']).mean()

    if min_threshold is not None:
        keep = mean_values[mean_values > min_threshold].index.to_list()
        timeseries_data = \
            timeseries_data.loc[timeseries_data['worm_index'].isin(keep), :]

    if max_threshold is not None:
        keep = mean_values[mean_values < max_threshold].index.to_list()
        timeseries_data = \
            timeseries_data.loc[timeseries_data['worm_index'].isin(keep), :]

    return timeseries_data

def filter_trajectories(
        timeseries_data, blob_features,
        min_traj_length=None, time_units=None,
        min_distance_traveled=None, distance_units=None,
        timeseries_names=None, min_thresholds=None,
        max_thresholds=None, units=None):

    if (min_traj_length is not None) and (min_traj_length>0):
        timeseries_data = _traj_lenght(timeseries_data, min_traj_length)
        blob_features =  blob_features.loc[timeseries_data.index, :]

    if min_distance_traveled is not None:
        timeseries_data = _distance_traveled(
            timeseries_data, min_distance_traveled)
        blob_features =  blob_features.loc[timeseries_data.index, :]

    if timeseries_names is not None and len(timeseries_names)>0:
        for name, min_thres, max_thres in \
            zip(timeseries_names, min_thresholds, max_thresholds):
                timeseries_data = _by_timeseries_values(
                    timeseries_data, name, min_thres, max_thres)
                blob_features =  blob_features.loc[timeseries_data.index, :]

    return timeseries_data, blob_features
