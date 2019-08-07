include("./../src/DataGenerator.jl")
include("./../src/nn.jl")
include("./../src/Dataset.jl")
include("./../src/Optimization.jl")
include("./../src/Vizualize.jl")
include("./../src/PrintLog.jl")
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
using .PrintLog

function main(logfile,α,η,epochs,n_known_of_each)
  @printlog logfile #Make a logfile
  device = cpu
  debug = true
  σ = relu
  dt = 0.1
  batch_size = 200
  reg_batch_size = 1000
  laplace_mode = 1
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
  forward = Chain(
    Conv((3,3), 3=>16,pad=1),BatchNorm(16, σ),
    IdentitySkipConv(Chain(Conv((3,3), 16=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt,Conv((1,1),16=>32)),
    IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt),
    IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt),
    IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt),
    IdentitySkipConv(Chain(Conv((3,3), 32=>64,pad=1,stride=2),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt,Conv((1,1),32=>64,stride=2)),
    IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt),
    IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt),
    IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt),
    IdentitySkipConv(Chain(Conv((3,3), 64=>128,pad=1,stride=2),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt,Conv((1,1),64=>128,stride=2)),
    IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt),
    IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt),
    IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt),
    BatchNorm(128, σ),MeanPool((8,8)),x -> dropdims(x,dims=(1,2))) |> device

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

  best_epoch,loss,time_spent=Optimization.training!(forward,classify,optimizer,epochs,α,t,v,batch_size,reg_batch_size,batch_shuffle,laplace_mode,track_laplacian)

  y=forward(v.x)
  u=classify(y)
  cg=Data.maxprob(Tracker.data(u))
  acc_val = Statistics.mean(cg .== v.c)*100\
  @info("accuracy of validation data= ",acc_val)
  @noprintlog
end


#This runs the code for various cases, unfortunately it is very slow at the moment, especially the last step where it compute the forward on all the validation data takes forever.
#I think this last problem can be partially fixed by changing the network away from training mode at the end, might be faster then.
epoch = 50
η = 0.1
using Logging
for n_known_of_each in [2, 20, 50, 100, 200, 400]
  for α in [0, 0.5]
    logfile = string(α,"_",n_known_of_each,".log")
    io = open(logfile, "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
    main(logfile,α,η,epoch,n_known_of_each)
    flush(io)
    close(io)
  end
end
