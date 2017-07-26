--[[
    Summary: This module implements a synaptic scaling rule of the form
    y = ax

    where a is a vector with the dimensionality of x scaling each row of x 
    independently.

    Author: Alireza Goudarzi
    RIKEN Brain Science Institute
    alireza.goudarzi@riken.jp



]]

package = "SynapticScaling"
version = "1.0-0"
source = {
   url = "./",
   tag = "v1.0",
}
description = {
   summary = "Implements neuron-wise synaptic scaling.",
   detailed = [[
      This package extend nn.Module and implements neuron-wise synaptic scaling. 
   ]],
   homepage = "http://me.github.com/luafruits",
   license = "MIT/X11"
}
dependencies = {
   "lua >= 5.1, < 5.4",
   "torch >= 7.0",
   "nn >= 1.0",
   "dpnn = scm-1"
}


build = {
   type = "builtin",
   modules = {
   
    init = "ssinit.lua",
    SynapticScaling = "SynapticScaling.lua",
      
   },
   copy_directories = { "test" },
}