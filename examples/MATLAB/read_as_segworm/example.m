%features_file = '/Volumes/behavgenom_archive$/Avelino/DEMO/Results/double_pick_200117/N2_N10_F1-3_Set2_Pos5_Ch5_20012017_153643_features.hdf5';
features_file = 'unc-9 (e101) on food L_2011_09_13__11_20___3___1_features.hdf5';

%% read data from the features hdf5 file.
[features_t, features_means, experiment_info] = readFeatureTable(features_file); %read data
[features, conversions] = convert2Segworm(features_t); %reformat data into a segworm-like structure

%% plot skeletons
delta_t = 10;
good = ~isnan(features.worm_1.posture.skeleton.x(1,:));
xx = features.worm_1.posture.skeleton.x(:, good);
yy = features.worm_1.posture.skeleton.y(:, good);

figure()
plot(xx, yy)

%% plot midbody velocity
figure()
xx = features.worm_1.extra.timestamp;
yy = features.worm_1.locomotion.velocity.midbody.speed;
plot(xx, yy)
xlabel('time (s)')
ylabel('velocity (um/s)')




