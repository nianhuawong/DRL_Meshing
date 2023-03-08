clear;clc;close all;
%% 环境、状态及动作定义
env = mesh_DRL_Action;
obsInfo = getObservationInfo(env); 
actInfo = getActionInfo(env);
rng(now)
%% 建立critic网络，DQN和DDPG将观察值state和动作值action同时作为Critic输入
hiddenLayerSize1 = 400;
hiddenLayerSize2 = 300; 

imgPath = [imageInputLayer(obsInfo(1).Dimension, 'Normalization', 'none', 'Name', 'image')
    convolution2dLayer(10, 2,'Name','conv1','Stride', 5, 'Padding',0)
    reluLayer('Name','relu1')
    fullyConnectedLayer(2,'Name','fc1')
    concatenationLayer(3,2,'Name','cat1')
    fullyConnectedLayer(hiddenLayerSize1,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(hiddenLayerSize2,'Name','fc3')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];
         
statePath  = [imageInputLayer(obsInfo(2).Dimension, 'Normalization', 'none', 'Name', 'state')
%               fullyConnectedLayer(L,'Name','state_FC1')
%               reluLayer('Name','state_Relu1')
              fullyConnectedLayer(1,'Name','fc5','BiasLearnRateFactor',0,'Bias',0)
             ];         
         
actionPath = [imageInputLayer(actInfo.Dimension, 'Normalization', 'none', 'Name', 'action')
              fullyConnectedLayer(hiddenLayerSize2,'Name','fc6','BiasLearnRateFactor',0,'Bias',zeros(hiddenLayerSize2,1))
             ];
          
criticNetwork = layerGraph(imgPath);
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = connectLayers(criticNetwork,'fc5','cat1/in2');
criticNetwork = connectLayers(criticNetwork,'fc6','add/in2');
plot(criticNetwork)

criticOpts = rlRepresentationOptions('LearnRate', 5e-3, 'GradientThreshold', 1);
% criticOptions.UseDevice = 'gpu';

critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'image', 'state'},'Action',{'action'}, criticOpts);

%% 建立actor网络，将观察state作为输入
imgPath = [
    imageInputLayer(obsInfo(1).Dimension,'Normalization','none','Name',obsInfo(1).Name)
    convolution2dLayer(10,2,'Name','conv1','Stride',5,'Padding',0)
    reluLayer('Name','relu1')
    fullyConnectedLayer(2,'Name','fc1')
    concatenationLayer(3,2,'Name','cat1')
    fullyConnectedLayer(hiddenLayerSize1,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(hiddenLayerSize2,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(2,'Name','fc4')
    tanhLayer('Name','tanh1')
    scalingLayer('Name','scale1','Scale',max(actInfo.UpperLimit))
    ];
actionPath = [
    imageInputLayer(obsInfo(2).Dimension,'Normalization','none','Name',obsInfo(2).Name)
    fullyConnectedLayer(1,'Name','fc5','BiasLearnRateFactor',0,'Bias',0)
    ];

actorNetwork = layerGraph(imgPath);
actorNetwork = addLayers(actorNetwork,actionPath);
actorNetwork = connectLayers(actorNetwork,'fc5','cat1/in2');
plot(actorNetwork)

actorOpts = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
% actorOptions.UseDevice = 'gpu';

actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo,...
    'Observation',{'image' 'state'}, 'Action', {'scale1'}, actorOpts);

%% 建立智能体DDPG agent
agentOpts = rlDDPGAgentOptions(...   
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',64,...
    'SampleTime', 1);
agentOpts.NoiseOptions.Variance = 100;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-6;

agent = rlDDPGAgent(actor,critic,agentOpts);

%% 训练智能体
averQuality = 0.9;
steps = 100;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',10000,...
    'MaxStepsPerEpisode',steps,...
    'Verbose',true,...
    'Plots','none',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',averQuality * steps,...
    'ScoreAveragingWindowLength',10);  %     "Plots", "training-progress"
%     "UseParallel","true")

%% 是否加载预训练的agent
loadAgent = false;
if loadAgent
    load('./agent/finalAgent_9.mat','agent');
end

%% 是否训练agent
doTraining = true;
if doTraining  
    trainingStats = train(agent,env,trainOpts);
    save("./agent/finalAgent_"+num2str(steps)+".mat",'agent')
end

%% 部署智能体
% simOptions = rlSimulationOptions('MaxSteps',500);
% experience = sim(env,agent,simOptions);
% totalReward = sum(experience.Reward)
