--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require 'datasets/transforms'

if #arg < 2 then
   io.stderr:write('Usage: th classify.lua [MODEL] [FILE]...\n')
   os.exit(1)
end

local ffi = require 'ffi'
-- find Images
local function findImages(dir)
   local imagePath = torch.CharTensor()

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
   --local imageClasses = {}

   while true do
      local line = f:read('*line')
      if not line then break end

      local filename = paths.basename(line)
      local path = paths.dirname(line) .. '/' .. filename

      table.insert(imagePaths, path)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   return imagePaths
end


-- Load the model
local model = torch.load(arg[1])

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.TenCrop(224),
}

local features
local N = 5

function string_output(output)
   local string_print = ''
   for i = 1, 38 do
      string_print = string_print .. ', ' .. output[i]
   end
   return string_print
end

all_test_paths = findImages(arg[2])

print('filename,c_0,c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11,c_12,c_13,c_14,c_15,c_16,c_17,c_18,c_19,c_20,c_21,c_22,c_23,c_24,c_25,c_26,c_27,c_28,c_29,c_30,c_31,c_32,c_33,c_34,c_35,c_36,c_37')
for _,imgpath in ipairs(all_test_paths) do
   -- load the image as a RGB float tensor with values 0..1
   local img = image.load(imgpath, 3, 'float')
   local name = paths.basename(imgpath)
   
   -- Scale, normalize, and ten crop the image
   -- View as mini-batch of size 10
   img_batch = transform(img)

   -- Get the output of the softmax and mean it
   local output = model:forward(img_batch:cuda()):mean(1)[1]
   local string_out = string_output(output)

   print(name .. string_out)
end
