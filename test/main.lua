
require 'nn'

require 'optim'
require 'dpnn'
require  'sys'
dl = require 'dataload'
require 'SynapticScaling'
require 'mlutils'

-- parse command line arguments
cmd = torch.CmdLine();
cmd:option('-gaincontrol',0,'Local Synaptic Scaling (1) adaptive/contextual, (2) fixed')
cmd:option('-actfunc','relu','Activation function for post gain control relu, sigm, tanh')
cmd:option('-gainact','sigm','Activation function of the gain channel relu, sigm, tanh');
cmd:option('-lr',.2,'Learning rate')
cmd:option('-testscale',0,'Scale the test set by a random number between [0,1] for each epoch.')
cmd:option('-trainscale',0,'Scale the train set by a random number between [0,1] for each epoch.')
cmd:option('-gpu',1,'which gpu to use')
cmd:option('-smalldata',1,'Mnist Datasize')
cmd:option('-perminv',1,'permutation invariant')
cmdparams = cmd:parse(arg)



-- load dataset
if cmdparams.smalldata == 1 then 
    trainData = torch.load('../data/train_mnist.t7')
    testData = torch.load('../data/test_mnist.t7')
    
else 
    trainData , testData = dl.loadMNIST();
end

index = torch.randperm(trainData.inputs:size(2)):long();
trainData.inputs = trainData.inputs:index(2,index);
testData.inputs = testData.inputs:index(2,index);

-- set parameters
local lr = cmdparams.lr;
local gainControl = cmdparams.gaincontrol;
local inputsize = 784;
local numnode = 50;
local batchsize = 100;
local epochsize = 10000;
local maxepoch = 50;
local numexp =  10;
-- convert input

--activation function
func = nn.Tanh;
-- initial alpha
alpha = .5;
k = .1;

net = nn.Sequential()
    :add(nn.Convert())
    :add(nn.View(784))
    :add(nn.Linear(784,20))
    :add(nn.SynapticScaling(20))
    :add(func())
    :add(nn.Linear(20,20))
    :add(nn.SynapticScaling(20))
    :add(func())
    :add(nn.Linear(20,10))
    :add(nn.SynapticScaling(10))
    :add(nn.LogSoftMax())

net.modules[4].weight:fill(alpha)
net.modules[7].weight:fill(alpha)
net.modules[10].weight:fill(alpha)

net.modules[3].weight:div(alpha):mul(k);
net.modules[6].weight:div(alpha):mul(k);
net.modules[9].weight:div(alpha):mul(k);
-- criterion
CF = nn.ClassNLLCriterion()


-- classfication
cm = optim.ConfusionMatrix(10)

if cmdparams.gpu > 0 then 
    -- load gpu code
    require 'cutorch'
    require 'cunn'
    cutorch.getDevice(cmdparams.gpu);


    -- take to GPU
   
    net:cuda();
    CF:cuda();
    trainData.inputs = trainData.inputs:cuda();
    trainData.targets = trainData.targets:cuda();
    testData.inputs = testData.inputs:cuda();
    testData.targets = testData.targets:cuda();

end

params, gparams  = net:getParameters();

-- setu p log

   print(' <w1,a1>:'..string.format('<%3.2f,%3.2f>',mlutils.norm(net.modules[3].weight:t()):mean(),mlutils.norm(net.modules[4].weight):mean())
                  ..' <w2,a2>:'..string.format('<%3.2f,%3.2f>',mlutils.norm(net.modules[6].weight:t()):mean(),mlutils.norm(net.modules[7].weight):mean())
                  ..' <w3,a3>:'..string.format('<%3.2f,%3.2f>',mlutils.norm(net.modules[9].weight:t()):mean(),mlutils.norm(net.modules[10].weight):mean()));


print(net)
for epoch=1,maxepoch do 
    sys.tic();
    for j, input, target in trainData:sampleiter(batchsize,epochsize) do 
        targetout = target;
        net:zeroGradParameters();
        net:forward(input);
       
 
        loss = CF:forward(net.output,target);
        dloss = CF:backward(net.output,target);
        net:backward(input,dloss); 
       
        net:updateParameters(lr);
    end
    --print(loss)
    cm:zero();
    cm:batchAdd(net.output,targetout)
    cm:updateValids();
    acc = cm.totalValid;

    net:forward(testData.inputs)
    
    cm:zero();
    cm:batchAdd(net.output,testData.targets)
    cm:updateValids();
    tstacc = cm.totalValid;
    time = sys.toc();

    print('loss: '..string.format('%5.3f',loss)
                  ..' acc:'..string.format('%5.3f',acc)
                  ..' tstacc:'..string.format('%5.3f',tstacc)
                  ..' elapsed:'..string.format('%3.2f',time)
                  ..' <w1,a1>:'..string.format('<%3.2f,%3.2f>',mlutils.norm(net.modules[3].weight:t()):mean(),mlutils.norm(net.modules[4].weight):mean())
                  ..' <w2,a2>:'..string.format('<%3.2f,%3.2f>',mlutils.norm(net.modules[6].weight:t()):mean(),mlutils.norm(net.modules[7].weight):mean())
                  ..' <w3,a3>:'..string.format('<%3.2f,%3.2f>',mlutils.norm(net.modules[9].weight:t()):mean(),mlutils.norm(net.modules[10].weight):mean()));
end

