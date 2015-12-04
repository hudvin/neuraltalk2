require 'torch'
require 'nn'
require 'nngraph'
-- exotics
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderStub'

require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:option('-model','','path to model to evaluate')
-- Basic options
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
-- For evaluation on a folder of images:
-- For evaluation on MSCOCO images from some split:
cmd:option('-input_h5','','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json','','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)

opt.model = "/home/kontiki/Downloads/neuraltalk2/model_id1-501-1448236541.t7"

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model', 'seq_per_img'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
local vocab = checkpoint.vocab -- ix -> word mapping


-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.crit = nn.LanguageModelCriterion()
protos.lm:createClones() -- reconstruct clones inside the language model
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end


-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
function process(image_data)
  protos.cnn:evaluate()
  protos.lm:evaluate()
  -- fetch a batch of data
  local data = image_data
  data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
  -- forward the model to get loss
  local feats = protos.cnn:forward(data.images)
  -- forward the model to also get generated samples for each image
  local sample_opts = { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
  local seq = protos.lm:sample(feats, sample_opts)
  local sents = net_utils.decode_sequence(vocab, seq)
  local entry = {image_id = data.infos[1].id, caption = sents[1]}
  return entry.caption
end


function get_label(image_path)
  local image_data = load_image(image_path)
  local caption = process(image_data)
  return caption
end


local function isImage(f)
  local supportedExt = {'.jpg','.JPEG','.JPG','.png','.PNG','.ppm','.PPM'}
  for _,ext in pairs(supportedExt) do
    local _, end_idx =  f:find(ext)
    if end_idx and end_idx == f:len() then
      return true
    end
  end
 return false
end

root_dir = "images/"
for file in paths.files(root_dir) do
  if isImage(file) then
    full_path = root_dir ..  file
    caption = get_label(full_path)
    print(caption .. " for " .. full_path)
  end	
end

