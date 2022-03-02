clear;clc;close all;
%% 环境、状态及动作定义
env = mesh_DRL_Action;
obsInfo = getObservationInfo(env); 
actInfo = getActionInfo(env);
rng(0)
%% 建立critic网络，DQN和DDPG将观察值state和动作值action同时作为Critic输入
L = 64; % number of neurons
statePath = [imageInputLayer([obsInfo.Dimension(1) obsInfo.Dimension(2) 1], 'Normalization', 'none', 'Name', 'state')
             fullyConnectedLayer(L,'Name','CriticStateFC1')
             reluLayer('Name','CriticStateRelu1')
             fullyConnectedLayer(L,'Name','CriticStateFC2')];
actionPath = [imageInputLayer([actInfo.Dimension(1) actInfo.Dimension(2) 1], 'Normalization', 'none', 'Name', 'action')
              fullyConnectedLayer(L,'Name','CriticActionFC1')
              reluLayer('Name','CriticActionRelu1')
              fullyConnectedLayer(L,'Name','CriticActionFC2')
             ];
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
criticNetwork = connectLayers(criticNetwork,'CriticActionFC2','add/in2');
% plot(criticNetwork)

criticOpts = rlRepresentationOptions('LearnRate',5e-3,'GradientThreshold',1);
% criticOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold', 1);

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
    fullyConnectedLayer(L,'Name','ActorFC4')
    tanhLayer('Name','ActorTanhLayer')
    fullyConnectedLayer(actInfo.Dimension(1),'Name','actor')
    ];
% plot(layerGraph(actorNetwork))
actorOpts = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1);
% actorOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'}, 'Action', {'actor'}, actorOpts);

%% 建立智能体DDPG agent
agentOpts = rlDDPGAgentOptions(...   
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',64,...
    'SampleTime', 1);
agentOpts.NoiseOptions.Variance = 0.1;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-4;

agent = rlDDPGAgent(actor,critic,agentOpts);

averQuality = 0.7;
episode = 100;
%% 训练智能体
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',10000,...
    'MaxStepsPerEpisode',episode,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',averQuality * episode,...
    'ScoreAveragingWindowLength',10); 

loadAgent = false;
if loadAgent
    load('./agent/finalAgent_4.mat','agent');
end

doTraining = true;
if doTraining  
    trainingStats = train(agent,env,trainOpts);
    save("./agent/finalAgent"+num2str(episode)+".mat",'agent')
else
    % Load pretrained agent for the example.
    load('./agent/finalAgent_2.mat','agent');
end
%% 部署智能体
% simOptions = rlSimulationOptions('MaxSteps',500);
% experience = sim(env,agent,simOptions);
% totalReward = sum(experience.Reward)
