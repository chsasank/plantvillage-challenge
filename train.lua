require 'optim'
require 'xlua'
require 'nn'
require 'image'
local c = require 'trepl.colorize'

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
        weightDecay = opt.weight_decay
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
end


function Trainer:scheduler(epoch)
    --[[--
    Learning rate scheduler

    # Parameters
    epoch: int
        adjusts learning rate based on this.
    --]]--
    decay = math.floor((epoch - 1) / 30)
    return self.opt.learningRate* math.pow(0.1, decay)
end


function Trainer:train()
    --[[--
    Trains model for an epoch.
    --]]--
    self.optimState.learningRate = self:scheduler(self.nEpoch)    
    print('=> Training epoch # ' .. self.nEpoch)    
    self.model:training() -- Make model trainable. 

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local trainingLoss = 0
    local numBatches = 0
    local tic = torch.tic()
    local count = 0
    self.confusion:zero()

    local input, target, output

    -- Loop over batches
    for input_, target_ in self.dataGen:trainGenerator(self.batchSize) do 
        if self.opt.backend ~= 'nn' then
            input = input_:cuda(); target = target_:cuda()
        else
            input = input_; target = target_
        end

        xlua.progress(count+input:size(1), self.dataGen.nbTrainExamples)
        numBatches = numBatches + 1

    -- Forward pass
        output = self.model:forward(input)
        local loss = self.criterion:forward(output, target)
        self.confusion:batchAdd(output, target)

    -- Backward pass
        self.model:zeroGradParameters()
        local critGrad = self.criterion:backward(output, target)
        self.model:backward(input, critGrad)

        -- Updates
        local _,fs = optim.sgd(feval, self.params, self.optimState)
        trainingLoss = trainingLoss + fs[#fs]
        count = count + input:size(1)
    end

    
    -- Keep track of losses and accuracies
    self.confusion:updateValids()
    local trainAcc = self.confusion.totalValid*100
    print(('Train Loss: '..c.cyan'%.4f'..' Accuracy: '..c.cyan'%.2f'..' \t time: %.2f s'):format(trainingLoss/numBatches, trainAcc, torch.toc(tic)))
    -- self:saveImages(input, target, output)

    self.nEpoch = self.nEpoch + 1
    return trainingLoss/numBatches
end 


function Trainer:validate()
    --[[--
    Validate model for an epoch. Loads data too.
    --]]--
    print('==> Validating')
    self.model:evaluate() -- Convert model to evaluate mode

    local count = 0
    local tic = torch.tic()
    local valLoss = 0
    local numBatches = 0
    self.confusion:zero()

    local input, target, output
    
    for input_, target_ in self.dataGen:valGenerator(self.batchSize) do
        if self.opt.backend ~= 'nn' then
            input = input_:cuda(); target = target_:cuda()
        else
            input = input_; target = target_
        end

        xlua.progress(count+input:size(1), self.dataGen.nbValExamples)
        
        -- Forward pass
        output = self.model:forward(input)
        local loss = self.criterion:forward(output, target)
        self.confusion:batchAdd(output, target)
    
        valLoss = valLoss + loss
        count = count + input:size(1)
        numBatches = numBatches + 1
    end
    
    -- Keep track of losses and accuracies
    self.confusion:updateValids()
    local valAcc = self.confusion.totalValid*100
    print(('Validation Loss: '..c.cyan'%.4f'..' Accuracy: '..c.cyan'%.2f'..' \t time: %.2f s'):format(valLoss/numBatches, valAcc, torch.toc(tic)))
    
    return valLoss/numBatches
end

function Trainer:getModel()
  return self.model:clearState()
end
