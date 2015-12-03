local utils = require 'misc.utils'
require 'lfs'
require 'image'

local DataLoaderStub = torch.class('DataLoaderStub')

function DataLoaderStub:__init(image_path)
  print('DataLoaderRaw loading image from path: ', image_path)
  self.files = {}
  self.ids = {}
  
  table.insert(self.files, image_path)
  table.insert(self.ids, tostring(0)) -- just order them sequentially
      
end
--[[
  Returns a batch of data:
  - X (N,3,256,256) containing the images as uint8 ByteTensor
  - info table of length N, containing additional information
  The data is iterated linearly in order
--]]
function DataLoaderStub:getBatch()
  local batch_size = 1
  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
  local max_index = self.N
  local wrapped = false
  local infos = {}

  local img = image.load(self.files[1], 3, 'byte')
  img_batch_raw[1] = image.scale(img, 256, 256)
  
  local info_struct = {}
  info_struct.id = self.ids[1]
  info_struct.file_path = self.files[1]
  table.insert(infos, info_struct)

  local data = {}
  data.images = img_batch_raw
  data.bounds = {it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}
  data.infos = infos
  return data
end

