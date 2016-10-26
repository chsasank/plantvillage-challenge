require 'optim'
require 'xlua'
require 'nn'
require 'image'
local c = require 'trepl.colorize'
local pp = require 'pl.pretty'

log = require 'log'

local Trainer = torch.class 'Trainer'

function Trainer:__init(model, criterion, dataGen, opt)
    --[[--
    Initializes Trainer object.

    # Parameters
    model: torch model
        Model to be used.
    criterion: torch criterion
        Criterion to be used.
    data: Data object
        Used to get training and validation sets
    opt: table 
        Table with options like learning rate, momentum etc.

    --]]--
    self.model = model
    self.criterion = criterion
    self.opt = opt
    self.optimState = {
        learningRate = opt.learningRate or 0.01,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay
        }
    self.batchSize = opt.batchSize or 32
    self.nbClasses = opt.nbClasses or 38

    self.params, self.gradParams = model:getParameters()
    self.nEpoch = 1

    self.dataGen = dataGen
    self.confusion = optim.ConfusionMatrix(self.nbClasses)
    if self.opt.backend ~= 'nn' then
        require 'cudnn'
        require 'cunn'
    end
    
    
    log.outfile = 'logs/'.. self.opt.model .. '.log'
    log.info("Training started")
    log.info(pp.write(opt))
    self.step = 5
end


function Trainer:scheduler(epoch)
    --[[--
    Learning rate scheduler

    # Parameters
    epoch: int
        adjusts learning rate based on this.
    --]]--
    decay = math.floor((epoch - 1) / 12)
    return self.opt.learningRate* math.pow(0.1, decay)
end


function Trainer:runEpoch(tag)
    --[[--
    Trains/Validates model for an epoch.
    --]]--

    self.optimState.learningRate = self:scheduler(self.nEpoch)
    local iterator, maxSamples  
    if tag =='train' then
        print('=> Training epoch # ' .. self.nEpoch)    
        self.model:training() -- i.e, switch dropout and otherlayers to training mode.
        iterator = self.dataGen:trainGenerator(self.batchSize)
        maxSamples = self.dataGen.nbTrainExamples
    else
        print('==> Validating')
        self.model:evaluate()  -- i.e, switch dropout and otherlayers to evaluate/determinstic mode.
        iterator = self.dataGen:valGenerator(self.batchSize)
        maxSamples = self.dataGen.nbValExamples
    end
    

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local epLoss, epAcc = 0
    local numBatches = 0
    local count = 0
    self.confusion:zero()

    local tic = torch.tic() -- to keep track of time for an epoch, stars timer
    
    -- Loop over batches
    for input, target in iterator do 
        if self.opt.backend ~= 'nn' then
            input = input:cuda()
            target = target:cuda()
        end

        xlua.progress(count+input:size(1), maxSamples)
        numBatches = numBatches + 1
        count = count + input:size(1)

        -- Forward pass
        local output = self.model:forward(input)
        local loss = self.criterion:forward(output, target)

        -- keep track of losses and log
        epLoss = epLoss + loss
        self.confusion:batchAdd(output, target)
        if numBatches%self.step == 0 then
            log.debug(tag, numBatches..'/'..math.ceil(self.dataGen.nbTrainExamples/self.batchSize),
        loss)
        end 


        if tag == 'train' then
            -- Backward pass
            self.model:zeroGradParameters()
            local critGrad = self.criterion:backward(output, target)
            self.model:backward(input, critGrad)

            -- Updates
            local _,fs = optim.sgd(feval, self.params, self.optimState)
        end
    end

    -- Keep track of losses and accuracies
    epLoss = epLoss/numBatches
    self.confusion:updateValids()
    epAcc = self.confusion.totalValid*100

    local stringToPrint = (tag..' Loss: '..c.cyan'%.4f'..' Accuracy: '..c.cyan'%.2f '..' \t time: %.2f s'):format(
            epLoss, epAcc, torch.toc(tic)) 

    print(stringToPrint)
    log.info(stringToPrint)
    
    if tag == 'train' then
        self.nEpoch = self.nEpoch + 1
    end
    return epLoss
end 



function Trainer:train()
    --[[--
    Trains model for an epoch.
    --]]--
    return self:runEpoch('train')
end 


function Trainer:validate()
    --[[--
    Validate model for an epoch. Loads data too.
    --]]--
    return self:runEpoch('validate')
end

function Trainer:getModel()
  return self.model:clearState()
end
