%%  load data
data = load("sample_dat.mat");

trials = length(data.dat);
psth = zeros(size(data.dat(1).spikes));

%% part a

for i = 1:trials
    psth = psth + data.dat(i).spikes;
end

t = (1:400)*1e-3;
psth_smooth = zeros(53,400);

for i = 1:length(psth(:,1))
    neuron = psth(i,:);
    mdl = fitrgp(t',neuron');
    psth_smooth(i,:) = resubPredict(mdl)';
end

plot(psth(1,:)); hold on;
plot(psth_smooth(1,:))

% The PSTH estimated with the GP prior appears smoother and changes on a slower
% timescale than the raw PSTH.

%% part b

pca_raw = pca(psth);
pca_smooth = pca(psth_smooth);

figure();
plot3(pca_raw(:,1),pca_raw(:,2),pca_raw(:,3));

figure();
plot3(pca_smooth(:,1),pca_smooth(:,2),pca_smooth(:,3))

% The raw PCs look very noisy and almost random. The GP PC
% trajectory is much smoother.

%% part c
kernSD = 30;

gpfa_traj = neuralTraj(0, data.dat);

[estParams, seqTrain] = postprocess(gpfa_traj, 'kernSD', kernSD);

plot3D(seqTrain, 'xorth', 'dimsToPlot', 1:3);

% The single trial trajectories seem to follow the same general curved path
% but individual trials have transient deviations from this path.

%% part d

% The observed deviation in the trajectories might correspond to
% differences in hand trajectory for different reach conditions for each trial, 
% or differences in hand trajectory for different trials of the same
% condition. Alternatively, the deviations may be a result of spiking
% noise since we don't know the true firing rate of the neurons.

% Given more data about the position of the monkey's hand, we might test the first hypothesis
% by seeing how well the deviations in the neural trajectories correlate with single-trial hand
% motion.




