include("./../src/DataGenerator.jl")
include("./../src/nn.jl")
include("./../src/Dataset.jl")
include("./../src/Optimization.jl")
include("./../src/Vizualize.jl")
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

function main(logfile,α,η,epochs,n_known_of_each,io)
  device = cpu
  debug = true
  σ = relu
  dt = 0.1
  batch_size = 200
  reg_batch_size = 500
  laplace_mode = 0
  batch_shuffle = true
  track_laplacian = false
  ntrain = 4000
  nval = 10000
  seed_number = 1234

  if debug
    Random.seed!(seed_number)
  end

  X_train,C_train,X_val,C_val = DataGenerators.cifar10(ntrain,nval,debug) |> device
  ik = DataGenerators.SelectKnownPoints(C_train,n_known_of_each=n_known_of_each)

  #Next load the data into the general structure that I want.
  d,t=Data.InitDataset("training",X_train,C_train,ik=ik,batch_size=batch_size,batch_shuffle=batch_shuffle)
  iik=Array{Int64,1}(undef,0)
  _,v=Data.InitDataset("validation",X_val,C_val,ik=iik,batch_size=batch_size,batch_shuffle=batch_shuffle)


  #Create the network - here we have WideNet28-2
  layer1 = Conv((3,3), 3=>16,pad=1)
  layer2 = Chain(BatchNorm(16, σ),IdentitySkipConv(Chain(Conv((3,3), 16=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt,Conv((1,1),16=>32)))
  layer3 = Chain(BatchNorm(32, σ),IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt))
  layer4 = Chain(BatchNorm(32, σ),IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt))
  layer5 = Chain(BatchNorm(32, σ),IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt))
  layer6 = Chain(BatchNorm(32, σ),IdentitySkipConv(Chain(Conv((3,3), 32=>64,pad=1,stride=2),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt,Conv((1,1),32=>64,stride=2)))
  layer7 = Chain(BatchNorm(64, σ),IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt))
  layer8 = Chain(BatchNorm(64, σ),IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt))
  layer9 = Chain(BatchNorm(64, σ),IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt))
  layer10 = Chain(BatchNorm(64, σ),IdentitySkipConv(Chain(Conv((3,3), 64=>128,pad=1,stride=2),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt,Conv((1,1),64=>128,stride=2)))
  layer11 = Chain(BatchNorm(128, σ),IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt))
  layer12 = Chain(BatchNorm(128, σ),IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt))
  layer13 = Chain(BatchNorm(128, σ),IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt))
  layer14 = Chain(BatchNorm(128, σ),MeanPool((8,8)),x -> dropdims(x,dims=(1,2)))
  #TODO consider how each layer gets regularization through velocity fields, does that make sense since they aren't pure residual layers, also some of them are not even close to residual layers, should these layers just not have regularization?
  #TODO perhaps create a variable that determines for each layer whether it should be regularized. Is it a problem if not all layers are regularized?
  forward = Chain(layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8,layer9,layer10,layer11,layer12,layer13,layer14) |> device

  classify = Chain(
    Dense(128,10),softmax) |> device

  loss(x,y) = sum(crossentropy(classify(forward(x)),y))
  ps=params(forward,classify)
  # optimizer = SGD(ps,η)
  optimizer = ADAM(ps,η)

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
  for i in partition(1:nval, 400)
    y=forward(v.x[..,i])
    u=classify(y)
    cg=Data.maxprob(Tracker.data(u))
    nhits =+ sum(maxprob(t.cp[:,i]) .== maxprob(Tracker.data(u)))
  end
  acc_val = nhits/nval*100
  @info("accuracy of validation data= ",acc_val)
end


#This runs the code for various cases, unfortunately it is very slow at the moment, especially the last step where it compute the forward on all the validation data takes forever.
#I think this last problem can be partially fixed by changing the network away from training mode at the end, might be faster then.
epoch = 50
η = 0.1
using Logging
for n_known_of_each in [2, 20, 50, 100, 200, 400]
  for α in [0.5,0]
    logfile = string(α,"_",n_known_of_each,".log")
    io = open(logfile, "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
    main(logfile,α,η,epoch,n_known_of_each,io)
    flush(io)
    close(io)
  end
end
