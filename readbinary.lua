require 'torch'

local arg = {...}
datafilename = arg[1]
print('loading '..datafilename)

function fromfile(fname)
   local file = io.open(fname .. '.type')
   local type = file:read('*all')

   local x
   if type == 'float32' then
      x = torch.FloatTensor(torch.FloatStorage(fname))
   elseif type == 'int32' then
      x = torch.IntTensor(torch.IntStorage(fname))
   elseif type == 'int64' then
      x = torch.LongTensor(torch.LongStorage(fname))
   else
      print(fname, type)
      assert(false)
   end
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   x = x:reshape(torch.LongStorage(dim))
   return x:float()
end

function synthetic_data(numLabs, timelength, numpeople)
   x = torch.Tensor(numLabs,numpeople,timelength)
   x[{{1},{},{}}] = torch.randn(1,numpeople,timelength)
   x[{{2},{},{}}] = torch.randn(1,numpeople,timelength)*2
   x[{{3},{},{}}] = x[{{2},{},{}}] + x[{{1},{},{}}]
   xmissing = torch.randn(numLabs,numpeople,timelength)
   xmissing[xmissing:lt(0.9)]=0.0
   x = torch.cmul(x,xmissing)
   return x
end

x = fromfile(datafilename)
return x

