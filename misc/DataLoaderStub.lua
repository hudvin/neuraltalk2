local utils = require 'misc.utils'
require 'lfs'
require 'image'


--[[
  Returns a batch of data:
  - X (N,3,256,256) containing the images as uint8 ByteTensor
  - info table of length N, containing additional information
  The data is iterated linearly in order
--]]
function load_image(image_path)
  local batch_size = 1
  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
  local wrapped = false
  local infos = {}

  local img = image.load(image_path, 3, 'byte')
  img_batch_raw[1] = image.scale(img, 256, 256)
  
  local info_struct = {}
  info_struct.id = 0 
  info_struct.file_path = image_path
  table.insert(infos, info_struct)

  local data = {}
  data.images = img_batch_raw
  data.infos = infos
  return data
end

