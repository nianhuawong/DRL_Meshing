clear;clc;close all;
env = mesh_DRL_Action;
obsInfo = getObservationInfo(env); 
actInfo = getActionInfo(env);
rng(0)
%%
% create a network to be used as underlying critic approximator
statePath = [imageInputLayer([obsInfo.Dimension(1) 1 1], 'Normalization', 'none', 'Name', 'state')
             fullyConnectedLayer(24,'Name','CriticStateFC1')
             reluLayer('Name','CriticRelu1')
             fullyConnectedLayer(24,'Name','CriticStateFC2')];
actionPath = [imageInputLayer([actInfo.Dimension(1) 1 1], 'Normalization', 'none', 'Name', 'action')
              fullyConnectedLayer(24,'Name','CriticActionFC1')];
commonPath = [additionLayer(2,'Name','add')
              reluLayer('Name','CriticCommonRelu')
              fullyConnectedLayer(1,'Name','output')];
          
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
% plot(criticNetwork)

% set some options for the critic
criticOpts = rlRepresentationOptions('LearnRate',5e-3,'GradientThreshold',1);

% create the critic based on the network approximator
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},criticOpts);
%%
actorNetwork = [
    imageInputLayer([4 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(24,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(2,'Name','Actor')
    ];
% plot(layerGraph(actorNetwork))
actorOpts = rlRepresentationOptions('LearnRate',8e-2,'GradientThreshold',1);

% actor = rlAbstractRepresentation(actorNetwork,obsInfo,actInfo,...
%     'Observation',{'state'}, 'Action', {'Actor'}, actorOpts);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'}, 'Action', {'Actor'}, actorOpts);

%%
% agentOpts = rlACAgentOptions(...
%     'NumStepsToLookAhead',32, ...
%     'DiscountFactor',0.99);

% agent = rlACAgent(actor,critic,agentOpts);

agentOpts = rlDDPGAgentOptions(...   
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',32);
agent = rlDDPGAgent(actor,critic,agentOpts);

%%
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',500,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480,...
    'ScoreAveragingWindowLength',10); 

trainingStats = train(agent,env,trainOpts);