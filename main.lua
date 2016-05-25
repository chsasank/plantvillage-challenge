require 'paths'
require 'optim'
require 'nn'

------------[[Specify your options here]]--------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 PlantVillage Challenge Training script')
cmd:text()
cmd:option('-learningRate',             0.01,       'initial learning rate for sgd')
cmd:option('-momentum',                 0.9,        'momentum term of sgd')
cmd:option('-maxEpochs',                250,        'Max # Epochs')
cmd:option('-batchSize',                32,         'batch size')
cmd:option('-nbClasses',                38,         '# of classes' )
cmd:option('-nbChannels',               3,          '# of channels' )
cmd:option('-backend',                  'cudnn',    'Options: cudnn | nn')
cmd:option('-retrain',                  'none',    	'Path to model to finetune')

local opt = cmd:parse(arg or {}) -- Table containing all these options


-----------[[Model and criterion here]]---------------
require 'models/alexnet.lua'
local net
if opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain);
    net = torch.load(opt.retrain)
else
    net = createModel(opt)
end

local criterion = nn.ClassNLLCriterion()

if opt.backend ~= 'nn' then
	require 'cunn'
	require 'cudnn'
	net = net:cuda()
	cudnn.convert(net, cudnn) --Convert the net to cudnn
	criterion = criterion:cuda()
end

require 'datasets/plantvillage.lua'
dgen = DataGen('datasets/crowdai/')

----[[Get your trainer object and start training]]-----
require 'train.lua'
local trainer = Trainer(net, criterion, dgen, opt)

local bestValAcc = 0

for n_epoch = 1,opt.maxEpochs do
    local trainAcc = trainer:train()  --Train on training set
    local valAcc = trainer:validate() --Valiate on valiadation set

    if n_epoch%25 == 0 then
        local save_path = paths.concat( opt.save, ('unet_epoch_%d'):format(n_epoch) )
        torch.save(save_path, net)
        print("Checkpointing Model")
    end

    if valAcc > bestValAcc then
        bestValAcc = valAcc
        print(('Current Best Validation Accuracy %.3f'):format(bestValAcc))
    end
    print("Epoch "..n_epoch.." complete")
end
