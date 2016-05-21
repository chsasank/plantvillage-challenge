require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'

------------[[Specify your options here]]--------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 PlantVillage Challenge Training script')
cmd:text()
cmd:option('-learningRate',             0.01,       'initial learning rate for adam')
cmd:option('-beta1',                    0.9,        'momentum term of adam')
cmd:option('-maxEpochs',                250,        'Max # Epochs')
cmd:option('-batchSize',                32,         'batch size')
cmd:option('-nbClasses',                38,         '# of classes' )
cmd:option('-nbChannels',               3,          '# of channels' )
cmd:option('-resume',                   'none',     'Path to directory containing checkpoint')
cmd:option('-backend',                  'cudnn',    'Options: cudnn | cunn')

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

net = net:cuda()
cudnn.convert(net, cudnn) --Convert the net to cudnn
criterion = criterion:cuda()


----[[Get your trainer object and start training]]-----
require 'train.lua'
