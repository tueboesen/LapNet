include("./../src/DataGenerator.jl")
include("./../src/nn.jl")
include("./../src/Dataset.jl")
include("./../src/Optimization.jl")
include("./../src/Vizualize.jl")
include("./../src/NeuralNetworks.jl")
import .LoadCustomLayers.IdentitySkip
import .LoadCustomLayers.IdentitySkipConv
import .LoadCustomLayers.MyDense
using .Optimization
using Flux
using Flux: crossentropy
using Flux.Tracker
using Random
using CuArrays
using .Data
using Base.Iterators: partition
using EllipsisNotation
using .NeuralNetworks

function main(logfile,α,η,epochs,n_known_of_each,io)
  device = gpu
  debug = true
  σ = relu
  dt = 0.1
  batch_size = 200
  reg_batch_size = 5
  laplace_mode = 0
  batch_shuffle = true
  track_laplacian = false
  ntrain = 6000
  nval = 10000
  seed_number = 1234

  if debug
    Random.seed!(seed_number)
  end

  X_train,C_train,X_val,C_val = DataGenerators.cifar10(ntrain,nval,debug) |> device
  ik = DataGenerators.SelectKnownPoints(C_train,n_known_of_each=n_known_of_each)
  #Next load the data into the general structure that I want.
  d,t=Data.InitDataset("training",X_train,C_train,ik=ik,batch_size=batch_size,batch_shuffle=batch_shuffle,device=device) |> device
  iik=Array{Int64,1}(undef,0)
  _,v=Data.InitDataset("validation",X_val,C_val,ik=iik,batch_size=batch_size,batch_shuffle=batch_shuffle,device=device) |> device

  #Create the network - here we have WideNet28-2
  nblocks = 3
  layers_pr_block = 4
  channels = [16,32,64,128]
  [forward,resnetlayers,classify]=WideNet(nblocks,layers_pr_block,channels,dt=dt,device=device)

  ps=params(forward,classify)
  loss(x,y) = sum(crossentropy(classify(forward(x)),y))
  # optimizer = ADAM(ps,η)
  optimizer = SGD(ps,η)

  nparams=sum(length(p) for p ∈ ps)

  #Report all relevant numbers to logfile
  @info("debug=",debug)
  @info("α=",α)
  @info("η=",η)
  @info("n_known_of_each=",n_known_of_each)
  @info("batch_size=",batch_size)
  @info("reg_batch_size=",reg_batch_size)
  @info("laplace_mode=",laplace_mode)
  @info("batch_shuffle=",batch_shuffle)
  @info("track_laplacian=",track_laplacian)
  @info("ntrain=",ntrain)
  @info("nval=",nval)
  @info("seed_number=",seed_number)
  @info("dt=",dt)
  @info("σ=",σ)
  @info("optimizer=",optimizer)
  @info("number of parameters = ",nparams)
  flush(io)

  best_epoch,loss,time_spent=Optimization.training!(forward,classify,optimizer,epochs,α,t,v,batch_size,reg_batch_size,batch_shuffle,laplace_mode,track_laplacian)

  #This last step need to be split up in smaller batches or the mode needs to be changed to testing instead of training. Right now it runs out of memory which is why it takes forever.
  #TODO put this in a clean function that runs tests.
  # model = Flux.mapleaves(Flux.data, model) this is how it is done according to https://discourse.julialang.org/t/untracking-a-flux-model/24811/2
  nhits = 0
  forward_fixed = Flux.mapleaves(Flux.data, forward) #Fix the parameters after training
  classify_fixed = Flux.mapleaves(Flux.data, classify) #Fix the parameters after training
  for i in partition(1:nval, 10000)
    y=forward_fixed(v.x[..,i])
    u=classify_fixed(y)
    # println("u",u)
    cg=Data.maxprob(u)
    nhits =+ sum(maxprob(v.cp[:,i]) .== cg)
  end
  acc_val = nhits/nval*100
  @info("accuracy of validation data= ",acc_val)
  println("accuracy of validation data= ",acc_val)
end

#This runs the code for various cases, unfortunately it is very slow at the moment, especially the last step where it compute the forward on all the validation data takes forever.
#I think this last problem can be partially fixed by changing the network away from training mode at the end, might be faster then.
# epoch = 20
# η = 0.1
# using Logging
# for n_known_of_each in [2, 20, 50, 100, 200, 400]
#   for α in [0]
#     logfile = string(α,"_",n_known_of_each,".log")
#     io = open(logfile, "w+")
#     logger = SimpleLogger(io)
#     global_logger(logger)
#     main(logfile,α,η,epoch,n_known_of_each,io)
#     flush(io)
#     close(io)
#   end
# end


#
# ch = [16, 32, 64, 128]
# nblocks = 3
# layers_pr_block = 4
# pixels_in = 32
# dt = 1
# layers=[]
# ResLayers = []
#
# push!(layers, Conv((3,3), 3=>channels[1],pad=1))
# push!(ResLayers,0)
# for i=1:nblocks
#     if i == 1
#         stride = 1
#     else
#         stride = 2
#     end
#     push!(layers, Chain(BatchNorm(ch[i], σ),IdentitySkipConv(Chain(Conv((3,3), ch[i]=>ch[i+1],pad=1,stride=stride),BatchNorm(ch[i+1], σ),Conv((3,3), ch[i+1]=>ch[i+1],pad=1)),dt,Conv((1,1), ch[i]=>ch[i+1],stride=stride))))
#     push!(ResLayers,1)
#     for j=2:layers_pr_block
#         push!(layers, Chain(BatchNorm(ch[i+1], σ),IdentitySkip(Chain(Conv((3,3), ch[i+1]=>ch[i+1],pad=1),BatchNorm(ch[i+1], σ),Conv((3,3), ch[i+1]=>ch[i+1],pad=1)),dt)))
#         push!(ResLayers,1)
#     end
# end
# pixels_out = Int(pixels_in/(2)^(nblocks-1))
# push!(layers, Chain(BatchNorm(ch[end], σ),MeanPool((pixels_out,pixels_out)),x -> dropdims(x,dims=(1,2))))
# push!(ResLayers, 0)
# function buildstring(layers)
#   layerstring = ""
#   for i=1:length(layers)
#       if i==length(layers)
#           layerstring = layerstring * "layers[" * string(i) * "]"
#       else
#           layerstring = layerstring * "layers[" * string(i) * "],"
#       end
#   end
#   return layerstring
# end
# layerstring=buildstring(layers)
# forward = eval(:(Chain(layerstring)))
dt = 0.1
device = cpu
nblocks = 3
layers_pr_block = 4
channels = [16,32,64,128]
forward,resnetlayers,classify=WideNet(nblocks,layers_pr_block,channels,dt=dt,device=device)
