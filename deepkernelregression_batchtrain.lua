require 'cunn'
require 'nn'
require 'torch'
require 'cutorch'
require 'gnuplot'
require 'paths'
require 'optim'
dofile('util.lua')
dofile('../deepDiagnosis/CDivTable_rebust.lua')

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)
local opt = lapp[[
	--gpuid           (default 1)
	--augment_time    (default 1)
	--flag_limit_gradient	 (default 1)
	--flag_control_bias (default 1)
	--save_train_network_dir (default 'networks_multivar_train')
	--save_valid_network_dir (default 'networks_multivar_validbest')
	--log_file (default 'log.txt')
	--test_baseline_gp (default 1)
	--test_baseline_kr (default 1)
	--evaluate_separately (default 1)
]]

augment_time = opt.augment_time
flag_limit_gradient = opt.flag_limit_gradient
flag_control_bias = opt.flag_control_bias
save_train_network_dir = opt.save_train_network_dir
save_valid_network_dir = opt.save_valid_network_dir
cutorch.setDevice(opt.gpuid)
torch.setnumthreads(1)
log_file = opt.log_file
test_baseline_gp = opt.test_baseline_gp
test_baseline_kr = opt.test_baseline_kr
evaluate_separately = opt.evaluate_separately
print(opt)

local args = {...}
mode = args[1]
datafilename = args[2]
save_train_network_dir = args[4]
save_valid_network_dir = args[5]
log_file = args[6]
if (#args == 7) then	
	initial_model_kernel_dir = args[7]
	print('loadign from '.. initial_model_kernel_dir)
	model_init = scandir(initial_model_kernel_dir..'lab'.. labix ..'_*.net')
	pretrained_kernel_layer = torch.load(model_init[1])
	print('initializing from model: '.. model_init[1])
	print(pretrained_kernel_layer)
end
print(args)

function init()
	os.execute('mkdir -p ' .. sys.dirname(log_file))
	log_file_open = io.open(log_file, "a")
	log_file_open:write(dump(args))
	log_file_open:write('\n-------- start multivar ' .. os.date("%x") ..' '.. os.date("%X")..' ------\n')

	learningRateKernelLayer = 0.001
	learningRateCombinationLayer = 0.005
	learningRate_list = {} --length of current model! this size has to come from somewhere. 
	sigma2 = 10
	basis = torch.range(-10,10)
	laplac_kernel = torch.exp(torch.div(torch.abs(basis), -1*sigma2))
	halfkwidth = math.floor(laplac_kernel:size(1)/2)
	max_grad = 100
	max_bias = 0.001
	gaussian_noise_var_x = 0.01
	gaussian_noise_var_t = 2
	learningRateDecay = 0.01
	trainIterations = 100
	peopleCountForTrain = 100000
	peopleCountForTrainSub = 1000
	peopleCountForValidate = 10000
	batchSize = 64
	batchSizeRegress = 256
	deep_layer_width={1}
	depth = 1
	l2_coef = 1

	x = assert(loadfile('readbinary.lua'))(datafilename)
	x = x[{{1,18},{},{}}]:clone()
	labcounts = x:size(1)
	timecounts = x:size(3)
	peoplecounts = x:size(2)
	print("number of people in training set:"..peoplecounts)
	batchSize = labcounts*4

	labels_all = {}
	labelsFile  = io.open('lab_labels.txt')
	local line = labelsFile:read("*l")
	while (line ~= nil) do
	   table.insert(labels_all,line)
	   line = labelsFile:read("*l")
	end
	labelsFile:close()

	model_preturb = nn.Parallel(2,3)
	for ix_preturb = 1, labcounts do
		model_preturb:add(nn.SpatialConvolutionMM(1, 1, 61, 1, 1, 1, 30, 0))
	end
	model_preturb = model_preturb:cuda()
end

function setup_network(labix, countX)
	log_file_open:write(labels_all[labix])
	log_file_open:write('\n')

	--print(labels_all)
	data = x[{{},{1,countX},{}}]:cuda()
	
	if (#args == 7) then
		pretrained_kernel_layer_weight = pretrained_kernel_layer:get(1):get(1).weight:view(100,21):clone()
		pretrained_kernel_layer_weight = pretrained_kernel_layer_weight * (0.1 / (pretrained_kernel_layer_weight:norm()))
	end
	
	kernel_layer = nn.Sequential()
	kernel_layer_1 = nn.ParallelTable()
	for ix = 1, labcounts do
		local lab_kernel_i = nn.Sequential()
		local lab_kernel_i_ratio = nn.ParallelTable()		
		local lab_kernel_i_top = nn.SpatialConvolutionMM(1, 1, halfkwidth*2+1, 1, 1, 1, 0, 0) --halfkwidth, 0)	
		lab_kernel_i_top.weight = lab_kernel_i_top.weight:fill(1) --laplac_kernel:clone():viewAs(lab_kernel_i_top.weight):clone() --
		lab_kernel_i_top.weight = lab_kernel_i_top.weight/(10*lab_kernel_i_top.weight:norm())
		lab_kernel_i_top.bias = lab_kernel_i_top.bias:fill(0)
		local lab_kernel_i_bottom = lab_kernel_i_top:clone('weight','bias')
		lab_kernel_i_ratio:add(lab_kernel_i_top)
		lab_kernel_i_ratio:add(lab_kernel_i_bottom)
		lab_kernel_i:add(lab_kernel_i_ratio)
		local divider_layer = CDivTable_robust()
		lab_kernel_i:add(divider_layer)
		kernel_layer_1:add(lab_kernel_i)
	end
	kernel_layer:add(kernel_layer_1)
	joiner_layer = nn.JoinTable(2,3)
	kernel_layer:add(joiner_layer)
	
	combination_layer = nn.Sequential()
	combination_layer_part1 = nn.ConcatTable()
	for ix = 1, labcounts do
		local indiv_model_i = nn.Sequential()
		indiv_model_i:add(nn.Reshape(1*labcounts*1))
		indiv_model_i:add(nn.Dropout())
		indiv_model_i:add(nn.Linear(labcounts*1, 1))
		indiv_model_i:get(3).weight = indiv_model_i:get(3).weight:fill(1)
		indiv_model_i:get(3).weight = indiv_model_i:get(3).weight/(10*indiv_model_i:get(3).weight:norm())
		indiv_model_i:get(3).bias = indiv_model_i:get(3).bias:fill(0)
		combination_layer_part1:add(indiv_model_i:clone())
	end
	combination_layer:add(combination_layer_part1)
	combination_layer:add(nn.JoinTable(1,1))

	kernel_layer = kernel_layer:cuda()
	combination_layer = combination_layer:cuda()

	for ix = 1, labcounts do
		kernel_layer:get(1):get(ix):get(1):get(2):share(kernel_layer:get(1):get(ix):get(1):get(1),'weight','bias') --,'gradWeight','gradBias'
	end

	criterion = nn.MSECriterion():cuda()
	dmsedf_table = {}
	mseloss_table = {}

	log_file_open:write('finished building model:')
	log_file_open:write('\n')
end

function load_network(labix, load_network_name_kernel_layer, load_network_name_combination_layer, countX)
	log_file_open:write(labels_all[labix])
	log_file_open:write('\n')

	print(labels_all[labix])
	data = x[{{},{1,countX},{}}]:cuda()
	kernel_layer = torch.load(load_network_name_kernel_layer)
	combination_layer = torch.load(load_network_name_combination_layer)
	
	log_file_open:write('finished loading model from ' .. load_network_name_combination_layer .. ' and '.. load_network_name_kernel_layer)
	log_file_open:write('\n')
end

function normalize(input)
	local inputnnx = input:ne(0):clone()
	local mean = torch.cdiv(input:sum(4):clone(), inputnnx:sum(4):clone()):squeeze() --size 18	
	mean[inputnnx:sum(4):clone():squeeze():abs():eq(0)] = 0.0	

	local std = torch.cdiv(torch.pow(input,2):clone():sum(4):clone(), inputnnx:sum(4):clone()):squeeze():clone() - torch.cmul(mean,mean):clone()
	std[inputnnx:sum(4):clone():squeeze():eq(0)] = 1.0 -- we dont' divide or mult by zero std. replace it by 1.0
	std[std:lt(0)] = 1.0 --sometimes it's -0.0 don't know why..
	std = torch.sqrt(std):clone()
	
	std = std:view(std:size(1),1):clone() -- size 18x1
	mean = mean:view(mean:size(1),1):clone() --size 18x1

	input = input - mean:repeatTensor(1, input:size(4)):clone()
	input = torch.cmul(input, inputnnx)
	stdtmp = std:clone() --:fill(1) --don't divide by std for now.. 
	stdtmp[stdtmp:lt(0.1)] = 1.0 --if std is small, don't divide and then don't multiply.
	input = torch.cdiv(input, stdtmp:repeatTensor(1,input:size(4)):clone()):clone()
	return input:clone(), inputnnx:clone(), mean:clone(), stdtmp:clone()
end

function normalize_target(target, mean, std)
	local targets_nnz = target:ne(0)
	target = target - mean
	if std > 0 then
		target = target/std
	end
	return torch.cmul(target,targets_nnz):clone()
end

function regress(data, model_kernel_layer, model_combination_layer)
	collectgarbage()
	local total_mse = 0
	local total_mse_counter = 0
	local splitor_module_top = nn.SplitTable(2,3):cuda()
	local splitor_module_bottom = nn.SplitTable(2,3):cuda()

	local tmp_store_weights_at_t = torch.CudaTensor(labcounts)
	model_kernel_layer:evaluate()
	model_combination_layer:evaluate()

	if (evaluate_separately) then
		-- we want to do leave=-one-out evaluation. So we make the kernel be blind to 
		-- the current value, and only use rest of the values to impute this. 
		-- this is achieved by zeroing out the kernel in the middle of it (for x=x')		
		for ix_bias = 1,labcounts do
			tmp_store_weights_at_t[ix_bias] = model_kernel_layer:get(1):get(ix_bias):get(1):get(1).weight[{{1},{halfkwidth+1}}]:squeeze() --
			model_kernel_layer:get(1):get(ix_bias):get(1):get(1).weight[{{1},{halfkwidth+1}}]:fill(0) --:get(1):get(1)
		end	
	end

	local batch_input = torch.CudaTensor(batchSizeRegress, 1, labcounts, timecounts)
	local batch_input_nnx = torch.CudaTensor(batchSizeRegress, 1, labcounts, timecounts)
	local batch_target = torch.CudaTensor(batchSizeRegress, 1, timecounts)
	local batch_mu_labix = torch.CudaTensor(batchSizeRegress,1, 1, 1)
	local batch_std_labix = torch.CudaTensor(batchSizeRegress,1, 1, 1)
	local batch_input_nnx_accross_labs = torch.CudaTensor(batchSizeRegress, 1, labcounts, timecounts):fill(0)
	local bix = 0

	for i = 1, data:size(2) do
		if data[{{labix},{i},{}}]:ne(0):sum() > 2 then
			local input = data[{{},{i},{}}]:clone():view(1,1,labcounts,timecounts):clone()			
			input, inputnnx, mean, std = normalize(input)			
			local target = data[{{labix},{i},{}}]

			bix = bix + 1
			batch_input[{{bix},{1},{},{}}] = input:clone()
			batch_input_nnx[{{bix},{1},{},{}}] = inputnnx:clone()
			batch_target[{{bix},{1},{}}] = target:clone()
			batch_mu_labix[{{bix},{1},{1},{1}}] = mean[labix]:squeeze()
			batch_std_labix[{{bix},{1},{1},{1}}] = std[labix]:squeeze()
			for t_regress = 1, timecounts do
				local t_hfWidth_or_1 = math.max(t_regress - halfkwidth, 1)
				local t_hfWidth_or_tcount = math.min(timecounts, t_regress + halfkwidth)				
				batch_input_nnx_accross_labs[{{bix},{1},{},{t_regress}}] = inputnnx[{{1},{1},{},{t_hfWidth_or_1, t_hfWidth_or_tcount}}]:clone():sum(4):clone():ne(0):clone()
			end

			if (bix == batchSizeRegress) then
				bix = 0				
				local inputtobigmodel = {}
				local o1 = splitor_module_top:forward(batch_input)
				local o2 = splitor_module_bottom:forward(batch_input_nnx)
				for o1ix, o1item in ipairs(o1) do
					table.insert(inputtobigmodel, {o1item:clone():view(batchSizeRegress,1,1,timecounts):clone(), o2[o1ix]:clone():view(batchSizeRegress,1,1,timecounts):clone()})
				end	

				local output_kernel_layer = model_kernel_layer:forward(inputtobigmodel)
				local output = model_combination_layer:forward(output_kernel_layer) --, batch_input_nnx_accross_labs}

				local results = torch.cmul(output, batch_std_labix:expand(output:size())) + batch_mu_labix:expand(output:size())
				local results_nnz = torch.cmul(results,batch_input_nnx[{{},{},{labix},{}}]:clone()):squeeze()
				local targets_nnz = batch_target:squeeze()
				total_mse = total_mse + torch.pow( results_nnz - targets_nnz, 2):sum()
				total_mse_counter = total_mse_counter + batch_input_nnx:sum()				
				-- print('rmse: '..math.sqrt(total_mse/total_mse_counter))
			end
		end
	end
	if (evaluate_separately) then
		-- restore the weights
		for ix_bias = 1,labcounts do			
			kernel_layer:get(1):get(ix_bias):get(1):get(1).weight[{{1},{halfkwidth+1}}]:fill(tmp_store_weights_at_t[ix_bias]) --:get(1):get(1)
		end	
	end

	log_file_open:write('regress finished')
	log_file_open:write(math.sqrt(total_mse/total_mse_counter))
	log_file_open:write('\n')
	print('regress:')
	print(math.sqrt(total_mse/total_mse_counter))
	return math.sqrt(total_mse/total_mse_counter)
end

function augment_input(input, t)
	local nnx = input:ne(0):clone()
	local gaussian_noise_vector = (torch.randn(input:size()):cuda() * gaussian_noise_var_x)
	local newinput = input:clone() + torch.cmul(nnx, gaussian_noise_vector)	

	if augment_time == 1 then		
		for labixx = 1, labcounts do
			for tix = t - halfkwidth,  t + halfkwidth do --this is for general regression, before and after
				if nnx[{{1},{1},{labixx},{tix}}]:squeeze() ~= 0 then
					local jump = math.floor((torch.randn(1) * gaussian_noise_var_t):squeeze())
					if jump ~= 0 and tix+jump > 1 and tix+jump < 2*halfkwidth+1 then
						local tmp_input = newinput[{{1},{1},{labixx},{tix+jump}}]:squeeze()
						newinput[{{1},{1},{labixx},{tix+jump}}] = input[{{1},{1},{labixx},{tix}}]:squeeze()
						newinput[{{1},{1},{labixx},{tix}}] = tmp_input
					end
				end			
			end		
		end
	end
	return newinput:clone()
end

function augment_input_faster(input)
	local nnx = input:ne(0)
	local gaussian_noise_vector = (torch.randn(input:size()):cuda() * gaussian_noise_var_x)
	local newinput = input + torch.cmul(nnx, gaussian_noise_vector)	

	if augment_time == 1 then
		for ix_preturb = 1, labcounts do
			local weight_preturb = torch.CudaTensor(1, 1, 61, 1, 1):fill(0):clone()	
			local jump = math.floor((torch.randn(1) * gaussian_noise_var_t):squeeze())
			if (math.abs(jump) < 30+1) then
				weight_preturb[{{1},{1},{31+jump},{1},{1}}]:fill(1)
			else
				weight_preturb[{{1},{1},{31},{1},{1}}]:fill(1)
			end
			model_preturb:get(ix_preturb).weight = weight_preturb:viewAs(model_preturb:get(ix_preturb).weight):clone()
		end		
		return (model_preturb:forward(newinput)):clone()
	end
	return newinput:clone()
end

function plot_weights()
	local weights_all = torch.CudaTensor(labcounts, labcounts, 2*halfkwidth+1)

	for lx = 1, labcounts do
		local weights_for_kernels_first_layer = combination_layer:get(1):get(lx):get(3).weight:clone():squeeze()
		-- print(weights_for_kernels_first_layer)
		for ix_weight = 1,labcounts do
			weights_all[{{lx},{ix_weight},{}}] = kernel_layer:get(1):get(ix_weight):get(1):get(1).weight:squeeze():viewAs(weights_all[{{lx},{ix_weight},{}}]):clone() * weights_for_kernels_first_layer[ix_weight]
		end	
	end
	gnuplot.figure(6)
	gnuplot.imagesc(weights_all:view(6,3,labcounts, 2*halfkwidth+1):clone():transpose(2,3):clone():view(6*labcounts, 3*(2*halfkwidth+1)):clone(), 'color')

	gnuplot.figure(5)
	gnuplot.plot({'train mse', torch.Tensor(table_mse_train)})
end


function train(maxEpoch, labix)
	local splitor_module_top = nn.SplitTable(2,3):cuda()
	local splitor_module_bottom = nn.SplitTable(2,3):cuda()
	table_mse_train = {0}

	for epoch = 1,maxEpoch do
		kernel_layer:training()
		combination_layer:training()
		
		-- kernel_params, kernel_params_gd = kernel_layer:getParameters()
		-- comb_params, comb_params_gd = combination_layer:getParameters()

		collectgarbage()		
		local validation_mini_data = data[{{},{1,512},{}}]		
		local total_mse = 0
		local total_mse_counter = 0

		print ('epoch'..epoch)
		plot_weights()
		local shuffled_ix = torch.randperm(peoplecounts)
		local shuffled_time = torch.randperm(timecounts)
		local shuffled_labs = torch.randperm(labcounts)
		local shuffled_timeix = torch.randperm(math.min(peopleCountForTrainSub , (peoplecounts * timecounts) - 1))

		local batch_input = torch.CudaTensor(batchSize, 1, labcounts,  2*halfkwidth +1):fill(0)
		local batch_input_nnx = torch.CudaTensor(batchSize, 1, labcounts,  2*halfkwidth +1):fill(0)
		local batch_target = torch.CudaTensor(batchSize, labcounts):fill(0)
		local batch_mu_labix = torch.CudaTensor(batchSize,1, 1, 1):fill(0)
		local batch_std_labix = torch.CudaTensor(batchSize,1, 1, 1):fill(0)
		local batch_input_nnx_accross_labs = torch.CudaTensor(batchSize, 1, labcounts, timecounts):fill(0)
		local bix = 0
		local batch_number = 0

		for oxx = 1, math.min(peopleCountForTrainSub , (peoplecounts * timecounts) - 1) do
			local ox = shuffled_timeix[oxx]
			local tmp1 = math.fmod(ox, timecounts); if tmp1 == 0 then; tmp1 = timecounts; end;
			local ix = math.floor(ox/(timecounts)) + 1
			-- local li = math.fmod(tmp1, labcounts)
			-- if li == 0 then
			-- 	li = labcounts
			-- end
			local tx = math.floor(tmp1)
			--local labix = shuffled_labs[li]
			local t = shuffled_time[tx]
			local t_hfWidth_or_1 = math.max(t - halfkwidth, 1)
			local t_hfWidth_or_tcount = math.min(timecounts, t + halfkwidth)
			local i = shuffled_ix[ix]
			for labix = 1, labcounts do
				if (i > 512) and (t > halfkwidth) and (t < timecounts - halfkwidth) and (data[labix][i][t] ~= 0) then
					local input = data[{{},{i},{t - halfkwidth, t + halfkwidth}}]:clone():view(1,1,labcounts, 2*halfkwidth +1):clone()							
					local target = torch.CudaTensor(1, labcounts):fill(0)
					target[{{1},{labix}}] = input[{{1},{1},{labix},{halfkwidth+1}}]:clone():squeeze()
					input[{{1},{1},{},{halfkwidth+1}}] = 0					
					input, inputnnx, mean, std = normalize(input)

					target = normalize_target(target, mean[labix][1], std[labix][1])
					input = augment_input(input, halfkwidth+1)

					bix = bix + 1
					batch_input[{{bix},{1},{},{}}] = input:clone()
					batch_input_nnx[{{bix},{1},{},{}}] = inputnnx:clone()
					batch_target[{{bix},{}}] = target:clone()
					batch_mu_labix[{{bix},{1},{1},{1}}] = mean[labix]:squeeze()
					batch_std_labix[{{bix},{1},{1},{1}}] = std[labix]:squeeze()

					-- backup_kernel_layer = kernel_layer:clone()
					-- backup_ombination_layer = combination_layer:clone()

					if (bix == batchSize) then
						collectgarbage()
						bix = 0	
						batch_number = batch_number + 1
						kernel_layer:zeroGradParameters()
						combination_layer:zeroGradParameters()
						local inputtobigmodel = {}				
						local o1 = splitor_module_top:forward(batch_input)
						local o2 = splitor_module_bottom:forward(batch_input_nnx)					
						for o1ix = 1, labcounts do
							table.insert(inputtobigmodel, {o1[o1ix]:clone():view(batchSize,1,1, 2*halfkwidth +1):clone(), o2[o1ix]:clone():view(batchSize,1,1, 2*halfkwidth +1):clone()})
						end
						local output_kernel_layer = kernel_layer:forward(inputtobigmodel)
						local output = combination_layer:forward(output_kernel_layer)
						local output_nnx = torch.cmul(output:clone(), batch_target:ne(0):clone())
						local mseloss = torch.CudaTensor(batchSize):fill(0)
						local msegd = torch.CudaTensor(batchSize, labcounts):fill(0)
						for ii = 1, batchSize do
							mseloss[{{ii}}] = criterion:forward(output_nnx[ii], batch_target[ii])
							msegd[{{ii},{}}] = criterion:backward(output_nnx[ii], batch_target[ii]):viewAs(msegd[{{ii},{}}]):clone()
						end

						msegd = msegd/batchSize
						if flag_limit_gradient == 1 then
							msegd = torch.clamp(msegd, -1*max_grad, max_grad)
						end
						table.insert(table_mse_train, mseloss:mean())
						print('+'..mseloss:mean())
						print('msegd:norm():'.. msegd:norm() .. ' LR:' .. learningRateCombinationLayer .. ' output norm:'.. output:norm() .. ' input norm:'.. input:norm() .. ' output_kr layer norm:'.. output_kernel_layer:norm() .. '   weights combination 1 norm: ' .. combination_layer:get(1):get(1):get(3).weight:norm())
						
						local combination_layer_gd = combination_layer:backward(output_kernel_layer, msegd)
						print('combination_layer_gd:norm():' .. combination_layer_gd:norm() .. ' LR:'.. learningRateKernelLayer .. '  weights kernel 1 norm:' .. kernel_layer:get(1):get(1):get(1):get(1).weight:norm())
						
						if flag_limit_gradient == 1 then
							combination_layer_gd = torch.clamp(combination_layer_gd, -1*max_grad, max_grad)
						end

						local input_gd = kernel_layer:backward(inputtobigmodel, combination_layer_gd)

						current_learning_rate_kernel = learningRateKernelLayer / (1 + epoch * learningRateDecay)
						current_learning_rate_combination = learningRateCombinationLayer / (1 + epoch * learningRateDecay)					
						
						combination_layer:updateParameters(current_learning_rate_combination)
						kernel_layer:updateParameters(current_learning_rate_kernel)
						
						if flag_control_bias == 1 then
							for ix_bias = 1, labcounts do
								torch.clamp(kernel_layer:get(1):get(ix_bias):get(1):get(2).bias, -1*max_bias, max_bias)
								kernel_layer:get(1):get(ix_bias):get(1):get(2):share(kernel_layer:get(1):get(ix_bias):get(1):get(1),'bias')
								kernel_layer:get(1):get(ix_bias):get(1):get(2):share(kernel_layer:get(1):get(ix_bias):get(1):get(1),'weight')
							end
						end
						
						-- plot_weights()
						-- if math.fmod(batch_number, 10) == 0 then
						-- 	old_validation_rmse = validation_rmse or 1000

						-- 	validation_rmse = regress(validation_mini_data, kernel_layer, combination_layer)					
						-- 	if old_validation_rmse < validation_rmse - 0.001 then
						-- 		print('lowering the learning rates and maximum allowed gradient by half')
						-- 		learningRateCombinationLayer = learningRateCombinationLayer * 0.5
						-- 		learningRateKernelLayer = learningRateKernelLayer * 0.5
						-- 		max_grad = max_grad * 0.5
						-- 		kernel_layer = backup_kernel_layer:clone()
						-- 		combination_layer = backup_ombination_layer:clone()
						-- 	end

						-- 	print('validation rmse: \n'..validation_rmse)
						-- 	log_file_open:write('mini-validation during training')
						-- 	log_file_open:write('\n')
						-- 	log_file_open:flush()
						-- end
						
						-- kernel_layer:training()
						-- combination_layer:training()
					end			
				end
			end
		end		
		log_file_open:write('training epoch mse' .. epoch)
		log_file_open:write(math.sqrt(total_mse/total_mse_counter))
		log_file_open:write('\n')
		
		local filename = paths.concat(save_train_network_dir .. '/kernel_layer_lab'.. labix ..'_epoch'..epoch ..'.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
		if paths.filep(filename) then
		  os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
		end
		print('Saving network to '..filename)
		torch.save(filename, kernel_layer)
		
		local filename = paths.concat(save_train_network_dir .. '/combination_layer_lab'.. labix ..'_epoch'..epoch ..'.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
		if paths.filep(filename) then
		  os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
		end
		print('Saving network to '..filename)	
		torch.save(filename, combination_layer)

	end
end

function plot_gd_batch_etc( batch_input, output_kernel_layer, output, msegd, combination_layer_gd, input_gd)
	-- print(batch_input:size())
	-- print(output_kernel_layer:size())
	-- print(combination_layer_gd:size())
	-- print(output:size())
	-- print(msegd:size())
	
	-- gnuplot.figure(1)
	-- gnuplot.imagesc(batch_input[{{1},{},{},{}}]:clone():view(labcounts, timecounts):float())
	torch.save('tmp1', batch_input)

	-- gnuplot.figure(2)
	-- gnuplot.imagesc(output_kernel_layer[{{1},{},{},{}}]:clone():view(labcounts, timecounts):float())
	torch.save('tmp2', output_kernel_layer)
	
	-- gnuplot.figure(3)
	-- gnuplot.imagesc(output[{{1},{},{},{}}]:clone():view(1, timecounts):float())
	torch.save('tmp3', output)

	-- gnuplot.figure(4)
	-- gnuplot.imagesc(msegd[{{1},{},{},{}}]:clone():view(1, timecounts):float())
	torch.save('tmp4', msegd)

	-- gnuplot.figure(5)
	-- gnuplot.imagesc(combination_layer_gd[{{1},{},{},{}}]:clone():view(labcounts, timecounts):float())
	torch.save('tmp5', combination_layer_gd)

	-- gnuplot.figure(5)
	-- gnuplot.imagesc(batch_target[{{1},{},{},{}}]:clone():view(1, timecounts):float())
	torch.save('tmp6', input_gd)
	
end

if mode == 'train' then
	init()
	log_file_open:write('-----------------train deep kernel reg------------------\n')
	setup_network(labix, peopleCountForTrain)
	train(trainIterations,labix)
	return 1
end

if mode == 'valid' then
	init()
	log_file_open:write('-----------------validate------------------\n')
	model_lists_kernel = scandir(save_train_network_dir..'/kernel_layer_lab'.. labix ..'_*.net')
	model_lists_combination = scandir(save_train_network_dir..'/combination_layer_lab'.. labix ..'_*.net')
	best_rmse = 1000
	best_rmse_ix = 0
	best_model_kernel = nil
	best_model_combination = nul
	for modelix, model_lists_item in ipairs(model_lists_kernel) do
		print (modelix .. ' ' .. model_lists_item)
		load_network(labix, model_lists_kernel[modelix], model_lists_combination[modelix], peopleCountForValidate)		
		rmse_i = regress(data, kernel_layer, combination_layer)
		if rmse_i < best_rmse then
			best_rmse = rmse_i
			best_rmse_ix = modelix			
			best_model_kernel = kernel_layer:clone()			
			best_model_combination = combination_layer:clone()
		end
		log_file_open:flush()
	end
	local best_valid_filename_kernel = save_valid_network_dir .. '/' .. paths.basename(model_lists_kernel[best_rmse_ix])
	os.execute('mkdir -p ' .. sys.dirname(best_valid_filename_kernel))
	if paths.filep(best_valid_filename_kernel) then
	  os.execute('mv ' .. best_valid_filename_kernel .. ' ' .. best_valid_filename_kernel .. '.old')
	end
	print('Saving network to '..best_valid_filename_kernel)
	log_file_open:write('Saving network to ')
	log_file_open:write(best_valid_filename_kernel)
	log_file_open:write('\n')
	torch.save(best_valid_filename_kernel, best_model_kernel)
	log_file_open:flush()

	local best_valid_filename_combination = save_valid_network_dir .. '/' .. paths.basename(model_lists_combination[best_rmse_ix])
	os.execute('mkdir -p ' .. sys.dirname(best_valid_filename_combination))
	if paths.filep(best_valid_filename_combination) then
	  os.execute('mv ' .. best_valid_filename_combination .. ' ' .. best_valid_filename_combination .. '.old')
	end
	print('Saving network to '..best_valid_filename_combination)
	log_file_open:write('Saving network to ')
	log_file_open:write(best_valid_filename_combination)
	log_file_open:write('\n')
	torch.save(best_valid_filename_combination, best_model_combination)
	log_file_open:flush()
end

if mode == 'test' then
	init()
	log_file_open:write('-----------------test------------------\n')
	model_lists_kernel = scandir(save_valid_network_dir..'/kernel_layer_lab'.. labix ..'_*.net')
	model_lists_combination = scandir(save_valid_network_dir..'/combination_layer_lab'.. labix ..'_*.net')
	
	for modelix, model_lists_item in ipairs(model_lists_kernel) do	
		print (modelix .. ' ' .. model_lists_item)	
		load_network(labix, model_lists_kernel[modelix], model_lists_combination[modelix], peopleCountForValidate)	
		rmse_i = regress(data, kernel_layer, combination_layer)		
		print(rmse_i)
		-- local w = big_model:get(1):get(1).weight
		-- local kernel_width =  math.floor(w:size(2)/labcounts)	
		-- gnuplot.splot('best kernel for '..labels_all[labix], w:float():view(labcounts,kernel_width):clone():squeeze(),'-')
	end
end

log_file_open:write('-----------------done------------------\n\n')
log_file_open:close()
