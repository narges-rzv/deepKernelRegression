require 'cunn'
require 'nn'
require 'torch'
require 'cutorch'
require 'gnuplot'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)
local opt = lapp[[
	--gpuid           (default 1)
	--augment_time    (default 1)
	--flag_limit_gradient	 (default 1)
	--flag_control_bias (default 1)
	--labix 		(default 1)
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
labix = args[3]
save_train_network_dir = args[4]
save_valid_network_dir = args[5]
log_file = args[6]
print(args)

function dump(o)
   if type(o) == 'table' then
      local s = '{ '
      for k,v in pairs(o) do
         if type(k) ~= 'number' then k = '"'..k..'"' end
         s = s .. '['..k..'] = ' .. dump(v) .. ','
      end
      return s .. '} '
   else
      return tostring(o)
   end
end

function  init()
	os.execute('mkdir -p ' .. sys.dirname(log_file))
	log_file_open = io.open(log_file, "a")
	log_file_open:write(dump(args))
	log_file_open:write('\n-------- start ------\n')

	learningRate = 0.01
	sigma2 = 10
	basis = torch.range(-10,10)
	normal_kernel = torch.exp(torch.div(torch.abs(basis), -1*sigma2))
	halfkwidth = math.floor(normal_kernel:size(1)/2)
	max_grad = 10
	max_bias = 10
	gaussian_noise_var_x = 0.05
	gaussian_noise_var_t = 1
	learningRateDecay = 0.01
	trainIterations = 100
	peopleCountForTrain = 10000
	peopleCountForValidate = 10000	
	batchSize = 100
	batchSizeRegress = 1000

	x = assert(loadfile('readbinary.lua'))(datafilename)
	labcounts = x:size(1)
	timecounts = x:size(3)
	peoplecounts = x:size(2)

	labels_all = {}
	labelsFile  = io.open('loinc_file.top1000.withLabels')
	local line = labelsFile:read("*l")
	while (line ~= nil) do
	   table.insert(labels_all,line)
	   line = labelsFile:read("*l")
	end
	labelsFile:close()
end

function covariance(x1)
	local xmax, dummy = x1:max(3)
	xmax = xmax:squeeze()
	local xmean = xmax:mean(2):squeeze()
	xmean = xmean:view(xmean:size(1),1)
	xmeanzero = xmax - xmean:expand(xmax:size(1),xmax:size(2))
	return torch.mm(xmeanzero,xmeanzero:t())	
end

function setup_network(labix, countX)
	log_file_open:write(labels_all[labix])
	log_file_open:write('\n')

	print(labels_all[labix])
	data = x[{{},{1,countX},{}}]:cuda()
	covmatrix = covariance(x)
	
	cov_row = covmatrix[{{},{labix}}] / covmatrix[labix][labix]
	kernel_matrix_init = torch.mm(cov_row,normal_kernel:view(1,normal_kernel:size(1))):fill(0.1)
	
	big_model = nn.Sequential()

	conv_ratio = nn.ParallelTable()
	conv_layer_top = nn.SpatialConvolutionMM(1,1,halfkwidth*2+1,covmatrix:size(1),1,1,halfkwidth,0)
	conv_layer_top.weight = kernel_matrix_init:viewAs(conv_layer_top.weight)
	conv_layer_top.bias = torch.Tensor({0}):viewAs(conv_layer_top.bias)
	conv_layer_clone_bott = conv_layer_top:clone('weight','bias')	
	conv_ratio:add(conv_layer_top)
	conv_ratio:add(conv_layer_clone_bott)

	big_model:add(conv_ratio)
	big_model:add(nn.CDivTable())

	big_model = big_model:cuda()
	conv_layer_clone_bott:share(conv_layer_top,'weight','bias')

	shift_network = nn.SpatialConvolutionMM(1,1,halfkwidth*2+1,covmatrix:size(1),1,1,halfkwidth,0):cuda()

	criterion = nn.MSECriterion():cuda()
	dmsedf_table = {}
	mseloss_table = {}

	log_file_open:write('finished building model:')
	log_file_open:write('\n')
end

function load_network(labix, load_network_name, countX)
	log_file_open:write(labels_all[labix])
	log_file_open:write('\n')

	print(labels_all[labix])
	data = x[{{},{1,countX},{}}]:cuda()
	big_model = torch.load(load_network_name)
	
	log_file_open:write('finished loading model from ' .. load_network_name)
	log_file_open:write('\n')
end

function normalize(input)
	local inputnnx = input:ne(0)
	local mean = torch.cdiv(input:sum(4),inputnnx:sum(4)):squeeze()
	mean[inputnnx:sum(4):squeeze():abs():eq(0)] = 0.0	

	local std = torch.cdiv(torch.pow(input,2):sum(4), inputnnx:sum(4)):squeeze() - torch.cmul(mean,mean)
	std[inputnnx:sum(4):squeeze():eq(0)] = 0.0
	std[std:lt(0)]=0.0
	std = torch.sqrt(std)
	
	std = std:view(std:size(1),1)
	mean = mean:view(mean:size(1),1)

	input = input - mean:expand(mean:size(1),input:size(4))
	input = torch.cmul(input, inputnnx)
	stdtmp = std:clone()
	stdtmp[stdtmp:eq(0)] = 1.0
	input = torch.cdiv(input, stdtmp:expand(stdtmp:size(1),input:size(4)))	
	return input:clone(), inputnnx:clone(), mean, std
end

function normalize_target(target, mean, std)
	target = target - mean
	if std > 0 then
		target = target/std
	end
	return torch.CudaTensor({target})
end

function regress(data, model)
	total_mse = 0
	total_mse_counter = 0

	if (evaluate_separately) then
		-- we want to do leave=-one-out evaluation. So we make the kernel be blind to 
		-- the current value, and only use rest of the values to impute this. 
		-- this is achieved by zeroing out the kernel in the middle of it (for x=x')
		local w = model:get(1):get(1).weight		
		local kernel_width =  math.floor(w:size(2)/labcounts)
		w:view(labcounts,kernel_width)[{{labix},{math.floor((kernel_width+1)/2)}}]:fill(0)		
	end

	batch_input = torch.CudaTensor(batchSizeRegress, 1, labcounts, timecounts)
	batch_input_nnx = torch.CudaTensor(batchSizeRegress, 1, labcounts, timecounts)
	batch_target = torch.CudaTensor(batchSizeRegress, 1, timecounts)
	batch_mu_labix = torch.CudaTensor(batchSizeRegress,1, 1, 1)
	batch_std_labix = torch.CudaTensor(batchSizeRegress,1, 1, 1)
	bix = 0

	for i = 1, peoplecounts do
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
			
			if (bix == batchSizeRegress) then
				bix = 0
				local output = model:forward({batch_input, batch_input_nnx})				
				local results = torch.cmul(output, batch_std_labix:expand(output:size())) + batch_mu_labix:expand(output:size())				

				local results_nnz = torch.cmul(results,batch_input_nnx[{{},{},{labix},{}}]:clone()):squeeze()
				local targets_nnz = batch_target:squeeze()
				total_mse = total_mse + torch.pow( results_nnz - targets_nnz, 2):sum()
				total_mse_counter = total_mse_counter + batch_input_nnx:sum()				
			end
		end
	end
	log_file_open:write('regress finished')
	log_file_open:write(math.sqrt(total_mse/total_mse_counter))
	log_file_open:write('\n')
	print('regress:')
	print(math.sqrt(total_mse/total_mse_counter))
	return math.sqrt(total_mse/total_mse_counter)
end

function limit_value(inputx, max_value)
	if inputx > max_value then
		inputx = max_value
	elseif inputx < -1*max_value then
		inputx = -1*max_value
	end
	return inputx
end

function augment_input(input, t)
	local nnx = input:ne(0)
	local gaussian_noise_vector = (torch.randn(input:size()):cuda() * gaussian_noise_var_x)
	local newinput = input + torch.cmul(nnx, gaussian_noise_vector)	

	if augment_time == 1 then		
		for labixx = 1, labcounts do
			nnxlab = nnx[{{},{},{labixx},{}}]:ne(0):squeeze() --1x169
			for tix = 1, timecounts do			
				if nnxlab[tix] == 1 and tix ~= t then
					local jump = math.floor((torch.randn(1) * gaussian_noise_var_t):squeeze())
					if jump ~= 0 and tix+jump > 1 and tix+jump < input:size(4) then
						local tmp_input = newinput[1][1][labixx][tix+jump]
						newinput[{{1},{1},{labixx},{tix+jump}}] = input[{{},{},{labixx},{tix}}]:squeeze()
						newinput[{{1},{1},{labixx},{tix}}]= tmp_input
					end
				end			
			end		
		end
	end
	return newinput:clone()
end

function train(maxEpoch, labix)
	big_model:training()
	for epoch = 1,maxEpoch do		
		collectgarbage()
		total_mse = 0
		total_mse_counter = 0
		print ('epoch'..epoch)
		print('bias'); print(conv_layer_top.bias); print(conv_layer_clone_bott.bias)		
		shuffled_ix = torch.randperm(data:size(2))
		shuffled_time = torch.randperm(data:size(3))
		--validScore = regress(data,big_model)
		for ox = 1, data:size(3)*data:size(2) - 1 do
			tx = math.fmod(ox,timecounts); if tx == 0 then; tx = timecounts; end;
			ix = math.floor(ox/timecounts) + 1
			t = shuffled_time[tx]		
			i = shuffled_ix[ix]			
			if data[labix][i][t] ~= 0 and data[{{labix},{i},{}}]:gt(0):sum() > 2 then
				big_model:zeroGradParameters()	

				local input = data[{{},{i},{}}]:clone():view(1,1,labcounts,timecounts):cuda()
				local target = input[1][1][labix][t]
				input[{{1},{1},{labix},{t}}]:fill(0)
				input = augment_input(input,t)

				input, inputnnx, mean, std = normalize(input)
				target = normalize_target(target, mean[labix][1], std[labix][1])				
		
				local output = big_model:forward({input, inputnnx})
				local mseloss = criterion:forward(output[{{1},{1},{1},{t}}], target)
				local msegd = criterion:backward(output[{{1},{1},{1},{t}}], target):squeeze()
				if flag_limit_gradient == 1 then
					msegd = limit_value(msegd, max_grad)					
				end
				backward_gradients = output:clone():zero()
				backward_gradients[{{1},{1},{1},{t}}]:fill(msegd)
				
				big_model:backward({input, inputnnx}, backward_gradients)

				current_learning_rate = learningRate / (1 + epoch * learningRateDecay)          
				big_model:updateParameters(current_learning_rate)
				if flag_control_bias == 1 then
					local tmp = limit_value(conv_layer_top.bias:squeeze(), max_bias)
					conv_layer_top.bias = torch.CudaTensor({tmp}):viewAs(conv_layer_top.bias)
					conv_layer_clone_bott:share(conv_layer_top,'bias')
				end				
				total_mse = mseloss + total_mse
				total_mse_counter = 1 + total_mse_counter			
			end		
		end
		gnuplot.figure(1)
		gnuplot.splot(conv_layer_top.weight:float():view(labcounts,normal_kernel:size(1)):clone())--,'color'		
		-- gnuplot.figure(2)
		-- gnuplot.plot(conv_layer_top.weight:float():view(labcounts,normal_kernel:size(1)):clone():abs():mean(1):squeeze(),'-')
		-- gnuplot.figure(3)
		-- gnuplot.plot(conv_layer_top.weight:float():view(labcounts,normal_kernel:size(1)):clone():mean(2):squeeze(),'-')
		print(total_mse/total_mse_counter)
		log_file_open:write('training epoch mse' .. epoch)
		log_file_open:write(math.sqrt(total_mse/total_mse_counter))
		log_file_open:write('\n')
		
		local filename = paths.concat(save_train_network_dir .. '/lab'.. labix ..'_epoch'..epoch ..'.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
		if paths.filep(filename) then
		  os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
		end
		print('Saving network to '..filename)
		torch.save(filename, big_model)

	end
end

if mode == 'train' then
	init()
	log_file_open:write('-----------------train------------------\n')
	setup_network(labix, peopleCountForTrain)
	train(trainIterations,labix)
	return 1
end

function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -a '..directory):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end

function mysplit(inputstr, sep)
   if sep == nil then
          sep = "%s"
   end
   local t={} ; i=1
   for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
          t[i] = str
          i = i + 1
   end
   return t
end

if mode == 'valid' then
	init()
	log_file_open:write('-----------------validate------------------\n')
	model_lists = scandir(save_train_network_dir..'/lab'.. labix ..'_*.net')
	best_rmse = 1000
	best_rmse_ix = 0
	best_model = nil
	for modelix, model_lists_item in ipairs(model_lists) do
		print (modelix .. ' ' .. model_lists_item)
		load_network(labix, model_lists_item, peopleCountForValidate)		
		rmse_i = regress(data, big_model)
		if rmse_i < best_rmse then
			best_rmse = rmse_i
			best_rmse_ix = modelix
			best_model = big_model:clone()			
		end
		log_file_open:flush()
	end
	local best_valid_filename = save_valid_network_dir .. '/' .. paths.basename(model_lists[best_rmse_ix])
	os.execute('mkdir -p ' .. sys.dirname(best_valid_filename))
	if paths.filep(best_valid_filename) then
	  os.execute('mv ' .. best_valid_filename .. ' ' .. best_valid_filename .. '.old')
	end
	print('Saving network to '..best_valid_filename)
	log_file_open:write('Saving network to ')
	log_file_open:write(best_valid_filename)
	log_file_open:write('\n')
	torch.save(best_valid_filename, best_model)
	log_file_open:flush()
end

if mode == 'test' then
	init()
	log_file_open:write('-----------------test------------------\n')
	model_lists = scandir(save_valid_network_dir..'/lab'.. labix ..'*.net')
	for modelix, model_lists_item in ipairs(model_lists) do	
		print (modelix .. ' ' .. model_lists_item)	
		load_network(labix, model_lists_item, peopleCountForValidate)

		rmse_i = regress(data, big_model)
		print(rmse_i)
	end
end

log_file_open:write('-----------------done------------------\n\n')
log_file_open:close()
