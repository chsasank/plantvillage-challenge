--[[
Loads from files with extensions 'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'
--]]

require 'paths'
local sys = require 'sys'
local ffi = require 'ffi'
t = require 'datasets/transforms.lua'

local function getClass(path)
    local className = paths.basename(paths.dirname(path))
    return tonumber(className:sub(3)) + 1
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}


local DataGen = torch.class 'DataGen'

function DataGen:__init(path)
   self.rootPath = path
   self.trainImgPaths = self.findImages(paths.concat(self.rootPath, 'train'))
   self.valImgPaths = self.findImages(paths.concat(self.rootPath, 'val'))
   self.nbTrainExamples = #self.trainImgPaths
   self.nbValExamples = #self.valImgPaths 
end


function DataGen:generator(pathsList, batchSize, preprocess) 
   batchSize = batchSize or 32
   
   local pathIndices = torch.randperm(#pathsList)
   local batches = pathIndices:split(batchSize)
   local i = 1
   local function iterator()
      if i <= #batches then
         local currentBatch = batches[i]         
         local imgList = {}
         local clsList = {}

         for j = 1, currentBatch:size(1) do
            local currentPath = pathsList[currentBatch[j]]
            local currentClass = getClass(currentPath)
            local ok, rawImg = pcall(function() return t.loadImage(currentPath) end)
             if ok then
                local procImg = preprocess(rawImg)
                table.insert(imgList, procImg)
                table.insert(clsList, currentClass)
            end
         end
         
        local X = torch.Tensor(#imgList, 3, 224, 224)
        local Y = torch.Tensor(#clsList)
        for j = 1, #imgList do
            X[j] = imgList[j]
            Y[j] = clsList[j]
        end
         
         i = i + 1
         return X, Y
      end
   end
   return iterator
end


function DataGen:trainGenerator(batchSize)
    local trainPreprocess = t.Compose{
        t.RandomSizedCrop(224),
        t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
        }),
        t.Lighting(0.1, pca.eigval, pca.eigvec),
        t.ColorNormalize(meanstd),
        t.HorizontalFlip(0.5),}

   return self:generator(self.trainImgPaths, batchSize, trainPreprocess)
end


function DataGen:valGenerator(batchSize)
    local valPreprocess = t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd),
         t.CenterCrop(224),}
   return self:generator(self.valImgPaths, batchSize, valPreprocess)
end


function DataGen.findImages(dir)
   --[[--
   Returns a table with all the image paths found in dir. Uses find.
   Following extensions are used to filter images: 'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'
   --]]--

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local filename = paths.basename(line)
      local path = paths.dirname(line).. '/' .. filename
      table.insert(imagePaths, path)
   end

   f:close()

   return imagePaths
end