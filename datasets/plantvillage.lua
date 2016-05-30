--[[
Loads from files with extensions 'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'
--]]

require 'paths'
local t = require 'datasets/transforms.lua'

local DataGen = torch.class 'DataGen'

function DataGen:__init(path)
    -- path is path of directory containing 'train' and 'val' folders
    -- find all the images in train and val folders. 
    self.rootPath = path
    self.trainImgPaths = self.findImages(paths.concat(self.rootPath, 'train'))
    self.valImgPaths = self.findImages(paths.concat(self.rootPath, 'val'))
    self.nbTrainExamples = #self.trainImgPaths
    self.nbValExamples = #self.valImgPaths 
    torch.setnumthreads(1) -- random hack to speed up loading images
end

local function getClass(path)
    -- gets class from the name of the parent directory
    local className = paths.basename(paths.dirname(path))
    return tonumber(className:sub(3)) + 1
end


function DataGen:generator(pathsList, batchSize, preprocess) 
    -- pathsList is table with paths of images to be iterated over
    -- batchSize is number of images to be loaded in one iteration
    -- preprocess is function which will be applied to image after it's loaded
    batchSize = batchSize or 32

    -- Split all the paths into random batches
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
            local ok, img = pcall(function() return t.loadImage(currentPath) end)
             if ok then
                if preprocess then img = preprocess(img) end
                table.insert(imgList, img)
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
        t.Lighting(0.1, t.pca.eigval, t.pca.eigvec),
        t.ColorNormalize(t.meanstd),
        t.HorizontalFlip(0.5),}

   return self:generator(self.trainImgPaths, batchSize, trainPreprocess)
end


function DataGen:valGenerator(batchSize)
    local valPreprocess = t.Compose{
         t.Scale(256),
         t.ColorNormalize(t.meanstd),
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
