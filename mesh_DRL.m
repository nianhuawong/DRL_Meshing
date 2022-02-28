clear;clc;close all;
% boundaryFile = './boundary_file.cas';
env = mesh_DRL_Action;
obsInfo = getObservationInfo(env); 
actInfo = getActionInfo(env);
rng(0)
criticNetwork = [
    imageInputLayer([4 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(1,'Name','CriticFC')];

criticOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);

critic = rlValueRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);

actorNetwork = [
    imageInputLayer([4 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(4,'Name','fc')
    ];

actorOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);

actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},actorOpts);

agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',32, ...
    'DiscountFactor',0.99);

agent = rlACAgent(actor,critic,agentOpts);


trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',500,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480,...
    'ScoreAveragingWindowLength',10); 

trainingStats = train(agent,env,trainOpts);