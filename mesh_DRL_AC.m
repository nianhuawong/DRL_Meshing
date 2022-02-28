clear;clc;close all;
%% 环境、状态及动作定义
env = mesh_DRL_Action;
obsInfo = getObservationInfo(env); 
actInfo = getActionInfo(env);
rng(0)
%% 建立critic网络
L = 128;
criticNetwork = [
    imageInputLayer([obsInfo.Dimension(1) obsInfo.Dimension(2) 1],'Normalization','none','Name','state')
    fullyConnectedLayer(L,'Name','CriticFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(1,'Name','CriticFC2')];

criticOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);

critic = rlValueRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);
% plot(layerGraph(criticNetwork))

%% 建立actor网络
actorNetwork = [
    imageInputLayer([obsInfo.Dimension(1) obsInfo.Dimension(2) 1],'Normalization','none','Name','state')
    fullyConnectedLayer(L,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(2*actInfo.Dimension(1),'Name','action')
    reluLayer('Name','ActorRelu2')];

actorOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);

actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},actorOpts);
% plot(layerGraph(actorNetwork))

%% 建立智能体AC agent
agentOpts = rlACAgentOptions('NumStepsToLookAhead',32,'DiscountFactor',0.99);
agent = rlACAgent(actor,critic,agentOpts);

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