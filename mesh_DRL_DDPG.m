clear;clc;close all;
%% 环境、状态及动作定义
env = mesh_DRL_Action;
obsInfo = getObservationInfo(env); 
actInfo = getActionInfo(env);
rng(0)
%% 建立critic网络，DQN和DDPG将观察值state和动作值action同时作为Critic输入
L = 32; % number of neurons
statePath = [imageInputLayer([obsInfo.Dimension(1) obsInfo.Dimension(2) 1], 'Normalization', 'none', 'Name', 'state')
             fullyConnectedLayer(L,'Name','CriticStateFC1')
             reluLayer('Name','CriticRelu1')
             fullyConnectedLayer(L,'Name','CriticStateFC2')];
actionPath = [imageInputLayer([actInfo.Dimension(1) 1 1], 'Normalization', 'none', 'Name', 'action')
              fullyConnectedLayer(L,'Name','CriticActionFC1')];
commonPath = [additionLayer(2,'Name','add')
              reluLayer('Name','CriticCommonRelu1')
              fullyConnectedLayer(L,'Name','CriticCommonFC1')
              reluLayer('Name','CriticCommonRelu2')
              fullyConnectedLayer(1,'Name','output')
              ];
          
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
% plot(criticNetwork)

% criticOpts = rlRepresentationOptions('LearnRate',5e-3,'GradientThreshold',1);
criticOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1);

critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},criticOpts);

%% 建立actor网络，将观察state作为输入
actorNetwork = [
    imageInputLayer([obsInfo.Dimension(1) obsInfo.Dimension(2) 1],'Normalization','none','Name','state')
    fullyConnectedLayer(L,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(L,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(L,'Name','ActorFC3')
    reluLayer('Name','ActorRelu3')
    fullyConnectedLayer(actInfo.Dimension(1),'Name','ActorFC4')
    tanhLayer('Name','Actor')
%     fullyConnectedLayer(actInfo.Dimension(1),'Name','Actor')
    ];
% plot(layerGraph(actorNetwork))
% actorOpts = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1);
actorOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'}, 'Action', {'Actor'}, actorOpts);

%% 建立智能体DDPG agent
agentOpts = rlDDPGAgentOptions(...   
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',32);
agentOpts.NoiseOptions.Variance = 0.6;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor,critic,agentOpts);

%% 训练智能体
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',10000,...
    'MaxStepsPerEpisode',1,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',1,...
    'ScoreAveragingWindowLength',10); 

trainingStats = train(agent,env,trainOpts);
%%
