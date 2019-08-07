using CuArrays
using Flux.Tracker
using Flux.Tracker: gradient
using Flux.Tracker: param, back!, grad
using Flux
using Flux.Tracker: update!
using Statistics
# using CuArrays
using Flux, Flux.Tracker, Flux.Optimise
using Metalhead, Images
using Metalhead: trainimgs
using Images.ImageCore
using Flux: onehotbatch, onecold
using Base.Iterators: partition

#Define my own layer

#create data
device = gpu

nx = 24
vec=range(-1,1,length=nx)
x=zeros(nx,nx)
y=zeros(nx,nx)
for i =1:nx
    global x
    x[i,:]=vec
    y[:,i]=vec
end
X=vcat(x[:]',y[:]',0*y[:]') |> device
labelss = zeros(nx,nx)
ii = (abs.(x) .< 0.75) .& (abs.(y) .< 0.75)
labelss[ii] .= 1
C = vcat(labelss[:]', 1 .- labelss[:]') |> device
train = ([(cat(X[:,i], dims = 2), C[:,i]) for i in partition(1:576, 576)])

#define network

ctn = 0
nonlinear = tanh
dt=0.1

m = Chain(
  IdentitySkip(MyDense(3, 3, nonlinear),dt),
  IdentitySkip(MyDense(3, 3, nonlinear),dt),
  IdentitySkip(MyDense(3, 3, nonlinear),dt),
  IdentitySkip(MyDense(3, 3, nonlinear),dt),
  IdentitySkip(MyDense(3, 3, nonlinear),dt),
  IdentitySkip(MyDense(3, 3, nonlinear),dt),
  IdentitySkip(MyDense(3, 3, nonlinear),dt),
  IdentitySkip(MyDense(3, 3, nonlinear),dt))|> device
m2 = Chain(
  Dense(3, 2),
  softmax) |> device

mtest1 = Chain(Dense(3, 3))
mtest2 = Chain(IdentitySkip(MyDense(3, 3, nonlinear),dt))
Flux.params(mtest1) |> length
Flux.params(mtest2) |> length

#define loss function
using Flux: crossentropy, Momentum

loss(x,y) = sum(crossentropy(m2(m(x)),y))
#define optimizer

# opt = Momentum(params(m,m2), η=0.01, ρ=0.9)
accuracy(x,y) = mean(onecold(m2(m(x)), 1:2) .== onecold(y,1:2))

#train network
# @time Flux.train!(loss, data, ADAM(0.1))


# d=train[1]
# x,y=d
# n=m2(m(x))
# crossentropy(n,y)
# l=loss(x,y)
λ = 0.5
epochs = 100
# opt = SGD(params(m,m2),λ)
# opt = Momentum(params(m,m2),λ)
opt = ADAM(params(m,m2),λ)
for i = 1:4
  global ctn
  # global λ /= 2
  for epoch = 1:epochs
    ctn += 1
    for d in train
      # print(d)
      # break
      l = loss(d...)
      print(ctn, " loss=",Flux.Tracker.data(l),"\n")
      # print("before",m[1],"\n")
      back!(l)
      opt()
      # print("after",m[1],"\n")
      break
      # @show accuracy(valX, valY)
    end
  end
end
#validate network
d=train[1]
xx,yy=d
n=Flux.Tracker.data(m(xx))
n_cpu=Flux.Tracker.data(m(xx)) |> cpu
n2=Flux.Tracker.data(m2(n))
lab=onecold(n2)


# using Plots
# gr() # We will continue onward using the GR backend
# # plotly()
# plot(x[labelss.==1],y[labelss.==1],seriestype=:scatter,title="Input data",layout = grid(2,2), leg=false)
# plot!(x[labelss.==0],y[labelss.==0],seriestype=:scatter)
#
# plot!(x[lab.==1],y[lab.==1],seriestype=:scatter,title="output data",subplot=2, leg=false)
# plot!(x[lab.==2],y[lab.==2],seriestype=:scatter,subplot=2, leg=false)
#
# plot!(n_cpu[1,lab.==1],n_cpu[2,lab.==1],n_cpu[3,lab.==1],seriestype=:scatter,title="3D visual",subplot=3, leg=false)
# plot!(n_cpu[1,lab.==2],n_cpu[2,lab.==2],n_cpu[3,lab.==2],seriestype=:scatter,subplot=3, leg=false)


using Plots
# gr() # We will continue onward using the GR backend
plotly()
plot(n_cpu[1,lab.==1],n_cpu[2,lab.==1],n_cpu[3,lab.==1],seriestype=:scatter,title="3D visual", leg=false)
plot!(n_cpu[1,lab.==2],n_cpu[2,lab.==2],n_cpu[3,lab.==2],seriestype=:scatter,leg=false)
