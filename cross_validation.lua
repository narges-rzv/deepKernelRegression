require 'cunn'
require 'nn'
require 'torch'
require 'cutorch'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)
local opt = lapp[[
	--labix		(default 1)
]]

labix = opt.labix

root = '../kernelregressiondata'
rootdata = '../'

save_train_network_dir_univar = root..'/networks_univar_train/'
save_train_network_dir_multivar = root..'/networks_multivar_train/'
save_valid_network_dir_univar = root..'/networks_univar_validbest/'
save_valid_network_dir_multivar = root..'/networks_multivar_validbest/'

train_data_torch_bin_filename = rootdata..'/data/train/train_lab_to_icd9_normalized'
valid_data_torch_bin_filename = rootdata..'/data/valid/valid_lab_to_icd9_normalized'
test_data_torch_bin_filename = rootdata..'/data/test/test_lab_to_icd9_normalized'

save_logfile_train = root..'/logfile_'..labix..'.log'
save_logfile_test = root..'/logfile_'..labix..'.log'
save_logfile_validate = root..'/logfile_'..labix..'.log'


train_flag=1
valid_flag=1
test_flag=1

--train uni and multikernelregression
if (train_flag == 1) then
	--assert(loadfile('kernelregression.lua'))('train', train_data_torch_bin_filename, labix, save_train_network_dir_univar, save_valid_network_dir_univar, save_logfile_train)
	assert(loadfile('multivarkernelregression.lua'))('train', train_data_torch_bin_filename, labix, save_train_network_dir_multivar, save_valid_network_dir_multivar, save_logfile_train)
end

if (valid_flag ==1) then
	--assert(loadfile("kernelregression.lua"))('valid', valid_data_torch_bin_filename, labix,save_train_network_dir_univar, save_valid_network_dir_univar, save_logfile_train)
	assert(loadfile("multivarkernelregression.lua"))('valid', valid_data_torch_bin_filename, labix, save_train_network_dir_multivar	, save_valid_network_dir_multivar, save_logfile_train)
end

if (test_flag==1) then
	assert(loadfile('kernelregression.lua'))('test',test_data_torch_bin_filename, labix, save_train_network_dir_univar, save_valid_network_dir_univar, save_logfile_train)
	assert(loadfile('multivarkernelregression.lua'))('test',test_data_torch_bin_filename, labix, save_train_network_dir_multivar	, save_valid_network_dir_multivar, save_logfile_train)
end












