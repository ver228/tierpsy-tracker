features_file = '../_data/N2 on food R_2010_01_11__11_37_05___4___1_features.hdf5';

%% read data from the features hdf5 file.
[features, features_means, experiment_info] = readFeatureTable(features_file);
features = convert2Segworm(features);

delta_t = 10;
good = ~isnan(features.worm_1.posture.skeleton.x(1,:));
xx = features.worm_1.posture.skeleton.x(:, good);
yy = features.worm_1.posture.skeleton.y(:, good);
figure()
plot(xx, yy)