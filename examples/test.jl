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

function main()
  device = cpu
  debug = true
  α = 0
  n=24           #nsamples = n^2
  η = 0.5        #Learning rate
  epochs = 50   #Number of epochs
  batch_size = 200
  reg_batch_size = 1000
  laplace_mode = 1
  batch_shuffle = true
  track_laplacian = false
  ntrain = 20000
  nval = 100
  n_known_of_each = 200

  if debug
    Random.seed!(1234)
  end

  X_train,C_train,X_val,C_val = DataGenerators.cifar10(ntrain,nval,debug) |> device
  ik = DataGenerators.SelectKnownPoints(C_train,n_known_of_each=2)

  # X_train,C_train=DataGenerators.squares(24)|> device
  # ik=collect(range(1,stop=size(X,2)))

  #Next load the data into the general structure that I want.
  d,t=Data.InitDataset(X_train,C_train,ik=ik,batch_size=batch_size,batch_shuffle=batch_shuffle)
  #Create the network
  #Load the network into the network structure
  σ = tanh
  dt = 0.1

  # forward = Chain(
  #   IdentitySkip(MyDense(3, 3, σ),dt),
  #   IdentitySkip(MyDense(3, 3, σ),dt),
  #   IdentitySkip(MyDense(3, 3, σ),dt),
  #   IdentitySkip(MyDense(3, 3, σ),dt),
  #   IdentitySkip(MyDense(3, 3, σ),dt),
  #   IdentitySkip(MyDense(3, 3, σ),dt),
  #   IdentitySkip(MyDense(3, 3, σ),dt)) |> device

  # classify = Chain(
  #   Dense(3, 2),
  #   softmax) |> device

  # forward = Chain(
  #   Conv((5,5), 3=>16, relu),
  #   x -> maxpool(x, PoolDims(x,(2,2))),
  #   Conv((5,5), 16=>8, relu),
  #   x -> maxpool(x, PoolDims(x,(2,2))),
  #   x -> reshape(x, :, size(x, 4)),
  #   Dense(200, 120),
  #   Dense(120, 84)) |> device
  conv1=Conv((3,3), 16=>32,pad=1)
  conv2=Conv((3,3), 32=>64,pad=1)
  conv3=Conv((3,3), 64=>128,pad=1)

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
  optimizer = SGD(params(forward,classify),η)
  # optimizer = ADAM(params(forward,classify),η)
  println(typeof(t.x))
  best_epoch,loss,time_spent=Optimization.training!(forward,classify,optimizer,epochs,α,t,t,batch_size,reg_batch_size,batch_shuffle,laplace_mode,track_laplacian)

  y=forward(t.x)
  u=classify(y)
  cg=Data.maxprob(Tracker.data(u))
  Vizualize.Plot(Tracker.data(y),getindex.(cg,1))
end
  # using Revise
main()
#
# y=forward(t.x)
# u=classify(y)
# _,cg=findmax(Tracker.data(u),dims=1)
# getindex.(cg, 1)
# cg[:]

# methods(Flux.params(forward), super=true)
# methodswith(typeof(Flux.params(forward)); supertypes=true)
# Flux.params(forward)[1]
# forward
# methodswith(typeof(forward); supertypes=true)
# methodswith(typeof(forward[1]); supertypes=true)
# forward[1]
# ConvDims

# forward = Chain(
#   Conv((3,3), 3=>16),BatchNorm(16, σ),
#   IdentitySkipConv((conv1,BatchNorm(32, σ),Conv((3,3), 32=>32)),dt,Conv((1,1),16=>32)),
#   IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
#   IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
#   IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
#   IdentitySkipConv((conv2,BatchNorm(64, σ),Conv((3,3), 64=>64)),dt,Conv((1,1),32=>64)),
#   IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
#   IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
#   IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
#   IdentitySkipConv((conv3,BatchNorm(128, σ),Conv((3,3), 128=>128)),dt,Conv((1,1),64=>128)),
#   IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
#   IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
#   IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
#   BatchNorm(128, σ),MeanPool((8,8))) |> device
