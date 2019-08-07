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
  epochs = 5   #Number of epochs
  batch_size = 100
  reg_batch_size = 20
  laplace_mode = 1
  batch_shuffle = true
  track_laplacian = false
  ntrain = 200
  nval = 100
  n_known_of_each = 3

  if debug
    Random.seed!(1234)
  end

  X_train,C_train,X_val,C_val = DataGenerators.cifar10(ntrain,nval,debug) |> device
  ik = DataGenerators.SelectKnownPoints(C_train,n_known_of_each=n_known_of_each)

  #Next load the data into the general structure that I want.
  d,t=Data.InitDataset("training",X_train,C_train,ik=ik,batch_size=batch_size,batch_shuffle=batch_shuffle)
  iik=Array{Int64,1}(undef,0)
  _,v=Data.InitDataset("validation",X_val,C_val,ik=iik,batch_size=batch_size,batch_shuffle=batch_shuffle)
  #Create the network
  σ = relu
  dt = 0.1


  forward = Chain(
    Conv((3,3), 3=>16,pad=1),BatchNorm(16, σ),
    IdentitySkipConv(Chain(Conv((3,3), 16=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt,Conv((1,1),16=>32)),
    BatchNorm(32, σ),MeanPool((32,32)),x -> dropdims(x,dims=(1,2))) |> device

  classify = Chain(
    Dense(32,10),softmax) |> device

  loss(x,y) = sum(crossentropy(classify(forward(x)),y))
  ps=params(forward,classify)
  optimizer = SGD(ps,η)
  # optimizer = ADAM(params(forward,classify),η)
  nparams=sum(length(p) for p ∈ ps)
  println("number of parameters = ",nparams)
  best_epoch,loss,time_spent=Optimization.training!(forward,classify,optimizer,epochs,α,t,v,batch_size,reg_batch_size,batch_shuffle,laplace_mode,track_laplacian)



  y=forward(v.x)
  u=classify(y)
  cg=Data.maxprob(Tracker.data(u))
  acc_val = Statistics.mean(cg .== v.c)*100\
  println("accuracy of validation data= ",acc_val)

end
  # using Revise
main()
