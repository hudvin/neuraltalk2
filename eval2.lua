require 'torch'
require 'nn'
require 'nngraph'
-- exotics
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'

require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

local utils = require 'misc.utils'
require 'lfs'
require 'image'



local NTPPrototype = torch.class('NTP')

function NTPPrototype:__init()
  cmd = torch.CmdLine()
  -- Basic options
  cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
  -- Sampling options
  cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
  cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
  cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
  cmd:option('-backend', 'cudnn', 'nn|cudnn')
  cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
  cmd:text()

  self.opt = cmd:parse(arg)
  self.model = "/home/kontiki/Downloads/neuraltalk2/model_id1-501-1448236541.t7"

  torch.manualSeed(self.opt.seed)
  torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

  
  if self.opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if self.opt.backend == 'cudnn' then require 'cudnn' end
    cutorch.manualSeed(self.opt.seed)
    cutorch.setDevice(self.opt.gpuid + 1) -- note +1 because lua is 1-indexed
  end

  local checkpoint = torch.load(self.model)
  local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model', 'seq_per_img'}
  for k,v in pairs(fetch) do 
    self.opt[v] = checkpoint.opt[v] -- copy over options from model
  end
  self.vocab = checkpoint.vocab -- ix -> word mapping

  -------------------------------------------------------------------------------
  -- Load the networks from model checkpoint
  -------------------------------------------------------------------------------
  self.protos = checkpoint.protos
  self.protos.expander = nn.FeatExpander(self.opt.seq_per_img)
  self.protos.crit = nn.LanguageModelCriterion()
  self.protos.lm:createClones() -- reconstruct clones inside the language model
  if self.opt.gpuid >= 0 then for k,v in pairs(self.protos) do v:cuda() end end
end

function NTPPrototype:process(image_data)
  self.protos.cnn:evaluate()
  self.protos.lm:evaluate()
  -- fetch a batch of data
  local data = image_data
  data.images = net_utils.prepro(data.images, false, self.opt.gpuid >= 0) -- preprocess in place, and don't augment
  -- forward the model to get loss
  local feats = self.protos.cnn:forward(data.images)
  -- forward the model to also get generated samples for each image
  local sample_opts = { sample_max = self.opt.sample_max, beam_size = self.opt.beam_size, temperature = self.opt.temperature }
  local seq = self.protos.lm:sample(feats, sample_opts)
  local sents = net_utils.decode_sequence(self.vocab, seq)
  local entry = {image_id = data.infos[1].id, caption = sents[1]}
  return entry.caption
end

function NTPPrototype:get_label(image_path)
  local image_data = self:load_image(image_path)
  local caption = self:process(image_data)
  return caption
end


function NTPPrototype:isImage(f)
  local supportedExt = {'.jpg','.JPEG','.JPG','.png','.PNG','.ppm','.PPM'}
  for _,ext in pairs(supportedExt) do
    local _, end_idx =  f:find(ext)
    if end_idx and end_idx == f:len() then
      return true
    end
  end
 return false
end

--[[
  Returns a batch of data:
  - X (N,3,256,256) containing the images as uint8 ByteTensor
  - info table of length N, containing additional information
  The data is iterated linearly in order
--]]
function NTPPrototype:load_image(image_path)
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




local ntp = NTP()
local caption = ntp:get_label("images/1.jpg")
print(caption)

root_dir = "images/"
for file in paths.files(root_dir) do
  if ntp:isImage(file) then
    full_path = root_dir ..  file
    caption = ntp:get_label(full_path)
    print(caption .. " for " .. full_path)
  end	
end

