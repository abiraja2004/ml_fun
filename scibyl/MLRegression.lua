require 'torch'
require 'nn'
require 'optim'

data3D = torch.Tensor{  
  {40,  6,  4},
  {44, 10,  4},
  {46, 12,  5},
  {48, 14,  7},
  {52, 16,  9},
  {58, 18, 12},
  {60, 22, 14},
  {68, 24, 20},
  {74, 26, 21},
  {80, 32, 24}
}

data2D = torch.Tensor{  
  {6,  4},
  {10,  4},
  {12,  5},
  {14,  7},
  {16,  9},
  {18, 12},
  {22, 14},
  {24, 20},
  {26, 21},
  {32, 24}
} 
function LinReg(num_params, num_outs, in_data)
  data = in_data
  data_size = #data
  
  -- define the model dimensionality
  ninputs = num_params
  noutputs = num_outs
  model = nn.Linear(ninputs, noutputs)
  
  -- define the loss function
  lossCriterion = nn.MSECriterion()
  
  if ninputs ~= data_size[2]-1 then
    return print("Please ensure the model input numbers match the data provided")
  end
  
  x, dl_dx = model:getParameters()
 
 local lossEval
 lossEval = function(xnew)
   if x ~= xnew then
    x:copy(xnew)
   end
   
   _idx_ = (_idx_ or 0) + 1
   if _idx_ > data_size[1] then
     _idx_ = 1
   end
   
   local sample = data[_idx_]
   local target = sample[{ {1} }]
   local inputs = sample[{ {2, ninputs+1} }]
   
   -- reset gradients, which get accumulated
   dl_dx:zero()
   
   -- eval loss function
   local loss_x = lossCriterion:forward(model:forward(inputs), target)
   -- now eval derivative of loss function wrt x at this point
   model:backward(inputs, lossCriterion:backward(model.output, target))
   
   return loss_x, dl_dx
 end
 
 sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
 }
 
 -- Now iterate some obscene amount of time overr the training data
 for i = 1, 1e4 do
   current_loss = 0
   
   for j = 1, data_size[1] do
     _, fs = optim.sgd(lossEval, x, sgd_params)
     current_loss = current_loss+fs[1]
   end
   
   current_loss = current_loss / data_size[1]
   print("Current Loss Function Average = " .. current_loss .. " at epoch = " .. i)
 end
 
 return model
end

LinReg(2, 1, data3D)