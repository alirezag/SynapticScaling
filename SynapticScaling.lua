--[[
    Summary: This module implements a synaptic scaling rule of the form
    y = ax

    where a is a vector with the dimensionality of x scaling each row of x 
    independently.

    Author: Alireza Goudarzi
    RIKEN Brain Science Institute
    alireza.goudarzi@riken.jp



]]
local SynapticScaling, parent = torch.class('nn.SynapticScaling', 'nn.Module')

function SynapticScaling:__init(inputSize)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(1, inputSize)
   self.gradWeight = torch.Tensor(1, inputSize)
   
   self:reset()
end



function SynapticScaling:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end
   return self
end



function SynapticScaling:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(2))
      self.output:cmul( self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.output:cmul( input, torch.expand(self.weight,input:size(1),self.weight:size(2)))
     -- if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function SynapticScaling:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
         self.gradInput:cmul( gradOutput, self.weight:expand(input:size(1),self.weight:size(2)))
    end
    return self.gradInput
end
  
function SynapticScaling:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight:add(scale*gradOutput:cmul(input):sum(1))
end

function SynapticScaling:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end

function SynapticScaling:clearState()
   return parent.clearState(self)
end

function SynapticScaling:__tostring__()
  return torch.type(self) ..
      string.format('(%d)', self.weight:size(2))
end