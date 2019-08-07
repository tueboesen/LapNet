function forward_w_hist(X,chain) # To do - This should overload Chain instead
   yl=[]
   y=chain[1](X)
   push!(yl,y)
   for i in range(2,length(chain))
     y=chain[i](y)
     push!(yl,y)
   end
   return yl
end

function training(forward,classify,optimizer,epochs,α,trainingset,validationset,batch_size,reg_batch_size,batch_shuffle,laplace_mode)
   t=trainingset
   v=validationset
   for epoch in range(epochs)
      if α != 0 #Compute regularization
           reg_batch, _ = Misc.sample_wo_repl!(vcat(t.ik,t.iu),reg_batch_size,batch_shuffle)
           ylreg = forward_w_hist(t.x[:,reg_batch])
           L0 = Regularization.Laplacian(t.x[:,reg_batch],0)
      end
      ik=copy(t.ik)
      while true #Loop through all known labels
           batch, ik = Misc.sample_wo_repl!(ik,min(reg_batch_size,length(ik)),batch_shuffle) # This will not work right now, what if batch is lar4ger than the samples
           if batch
               break
           end
           y = forward(t.x[:,batch])
           u = classify(y)
           misfit = sum(crossentropy(u, t.cp[batch]))
           if α != 0
               ylreg = forward_w_hist(t.x[:,reg_batch])
               if laplace_mode != 0
                   Ln = Regularization.Laplacian(t.x[:,reg_batch],0)
                   reg = Regularization.Regularization(ylreg,L0,Ln)
               else
                   reg = Regularization.Regularization(ylreg,L0)
               end
               reg = α * reg / length(reg_batch)
           end
           loss = misfit + reg
           back!(l)
           opt()
      println(epoch)
      end
   end
end


include("./../src/Misc.jl")
include("./../src/Regularization.jl")
import .Misc.sample_wo_repl
using .Regularization
v=t
epoch = 1
A=vcat(t.ik,t.iu)
reg_batch, _ = Misc.sample_wo_repl!(vcat(t.ik,t.iu),reg_batch_size,batch_shuffle)

function forward_w_hist(X,chain) # To do - This should overload Chain instead
   yl=[]
   y=chain[1](X)
   push!(yl,y)
   for i in range(2,length(chain))
     y=chain[i](y)
     push!(yl,y)
   end
   return yl
end

x=t.x[:,reg_batch]
A,ϵ = Regularization.AdjacencyMatrix(x)
L0 = Regularization.GraphLaplacian(x,A,ϵ)
ylreg = forward_w_hist(t.x[:,reg_batch],forward)

reg = 0
layer = ylreg[1]
# for layer in yl
using LinearAlgebra
reg += LinearAlgebra.tr(layer*L0*LinearAlgebra.transpose(layer))

LinearAlgebra.transpose(layer)

Reg = Regularization.Compute_Regularization(ylreg,L0)

# ylreg = forward_w_hist(t.x[:,reg_batch])
L0 = Regularization.Laplacian(t.x[:,reg_batch],0)
ik=copy(t.ik)
batch, ik = Misc.sample_wo_repl!(ik,min(reg_batch_size,length(ik)),batch_shuffle) # This will not work right now, what if batch is lar4ger than the samples
y = forward(t.x[:,batch])
u = classify(y)
t.cp[:,batch]
misfit = sum(crossentropy(u, t.cp[:,batch]))
if α != 0
    ylreg = forward_w_hist(t.x[:,reg_batch])
    if laplace_mode != 0
        Ln = Regularization.Laplacian(t.x[:,reg_batch],0)
        reg = Regularization.Regularization(ylreg,L0,Ln)
    else
        reg = Regularization.Regularization(ylreg,L0)
    end
    reg = α * reg / length(reg_batch)
end
loss = misfit + reg
back!(l)
opt()
println(epoch)
end
#
# for epoch = 1:epochs
#   for d in data
#     l = loss(d...)
#     print(epoch," loss=",Flux.Tracker.data(l),"\n")
#     back!(l)
#     opt()
#   end
# end


function AdjacencyMatrix_(x;knn=9)
  n=size(x,2)
  I=zeros(knn,n)
  J=zeros(knn,n)
  dd = zeros(n,knn)
  di = zeros(n,knn)
  for i=1:n
      r = x .- x[:,i]
      d = sum(r.*r,dims=1)
      idx = sortperm(d[:])
      di[i,:] = d[idx[1:knn]]
      I[:,i] = i.*ones(knn,1)
      J[:,i] = idx[1:knn]
  end
  A = SparseArrays.sparse(I[:],J[:],ones(length(J[:]),1)[:],n,n)
  A = A+A'  # Warning this is not a normal adjencymatrix, since we have values that are 2 instead of 1 in it as well!
  return A, Statistics.median(di)
end

function AdjacencyMatrix(x;knn=9,track=true)
  if track
      A, di = AdjacencyMatrix_(x,knn=knn)
  else
      A, di = AdjacencyMatrix_(Tracker.data(x),knn=knn)
  end
  return A, di
end
AdjacencyMatrix(X,track=true)






function unitvec(xs::AbstractVector{T}) where {T <: Real}
    S = typeof(√one(T))         # takes care of T ≡ Int64, etc
    n = length(xs)
    y = Vector{S}(undef, n + 1)
    r = one(S)
    for (i, x) in enumerate(xs)
        z = tanh(x)
        y[i] = z * √r
        r *= 1 - abs2(z)
    end
    y[end] = √r
    y
end

aa=Flux.param(ones(3))
ab=ones(3)
unitvec(ab)


device = gpu
a=[0.3 0.5 0.7; 0.2 0.4 0.8]

a|> device
a
typeof(a)

_,cg=findmax(a,dims=1)
c=getindex.(cg,1)

include("./../src/Regularization.jl")
include(".jl")
using .Regularization
Laplacian()
foo()
Regularization.foo()

CuArray <: AbstractArray
Float32 <: AbstractFloat
Int64 <: Signed


# in general you can find the source with
#
# using Flux; methods(sum, (Flux.TrackedArray, ))
# or, even better,
#
# edit(first(methods(sum, (Flux.TrackedArray, ))))
# will open it in your editor directly.

using Plots
function plottest()
    # gr(show = true)
    plotly(show = true)
    x=randn(10)
    y=randn(10)
    !plot(x,y)
    #display(p1)
    # gui(p1)
end
plottest()



using Flux: onehotbatch, onecold
using Metalhead, Images
using Metalhead: trainimgs
using Metalhead: valimgs
# The image will give us an idea of what we are dealing with.
# ![title](https://pytorch.org/tutorials/_images/cifar10.png)

function test()
    Metalhead.download(CIFAR10)
end
test()
ntrain=500
nval = 200
Metalhead.download(CIFAR10)
shuffles=true
getarray(X) = float.(permutedims(channelview(X), (2, 3, 1)))
if ntrain > 0
    train = trainimgs(CIFAR10)
    xx=collect(range(1,stop=length(train)))
    idx,_=sample_wo_repl!(xx,ntrain,shuffles)
    C_train=[train[i].ground_truth.class for i in idx]
    X = [getarray(train[i].img) for i in idx]
    # size(X[1])
    X_train = Array{Float64}(undef, 32,32,3,ntrain)
    for (i,xi) in enumerate(X)
        X_train[:,:,:,i] = xi
    end

end
tt=size(X[1])[:]
tt[1]
cc=getarray(train[1].img)
using RecursiveArrayTools
add_dim(x::Array) = reshape(x, (size(x)...,1))
a=X_train
permutedims(reshape(vcat(a...), (size(a[1]), length(a))))

vcat(a...)
length(a[1])
size(a[1])

size(X_train[1])
xx=add_dim(X_train)
Xxx=vcat(X_train...)
typeof(X_train)
typeof(Xxx)

if nval > 0
    val = valimgs(CIFAR10)
    xx=collect(range(1,stop=length(val)))
    idx,_=sample_wo_repl!(xx,ntrain,shuffle)
    C_val=[val[i].ground_truth.class for i in idx]
    X_val = [getarray(val[i].img) for i in idx]
end

function sample_wo_repl!(A,batch_size,shuffle)
    nmax=length(A)
    n=min(nmax,batch_size)
    sample = Array{eltype(A)}(undef,n)
    if shuffle
        for i in 1:n
            sample[i] = splice!(A, rand(eachindex(A)))
        end
    else
        for i in 1:n
            sample[i] = splice!(A, eachindex(A))
        end
    end
    points_remaining = A
    return sample,points_remaining
end




# labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)




image(x) = x.img # handy for use later
ground_truth(x) = x.ground_truth
image.(X[rand(1:end, 10)])

# The images are simply 32 X 32 matrices of numbers in 3 channels (R,G,B). We can now
# arrange them in batches of say, 1000 and keep a validation set to track our progress.
# This process is called minibatch learning, which is a popular method of training
# large neural networks. Rather that sending the entire dataset at once, we break it
# down into smaller chunks (called minibatches) that are typically chosen at random,
# and train only on them. It is shown to help with escaping
# [saddle points](https://en.wikipedia.org/wiki/Saddle_point).

# Defining a `getarray` function would help in converting the matrices to `Float` type.


# The first 49k images (in batches of 1000) will be our training set, and the rest is
# for validation. `partition` handily breaks down the set we give it in consecutive parts
# (1000 in this case). `cat` is a shorthand for concatenating multi-dimensional arrays along
# any dimension.
tt=(cat(imgs[1], dims = 4), labels[:,1])

# train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, 1000)])
train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:30, 5)])
valset = 49001:50000
valX = cat(imgs[valset]..., dims = 4) |> gpu
valY = labels[:, valset] |> gpu


struct test{T}
  name::String
  x :: AbstractArray{<: AbstractArray{<: AbstractFloat, T}}   #
  function test(name::String,x::AbstractArray{<: AbstractArray{<: AbstractFloat, T}}) where T
   new{T}(name,x)
  end
end
nn="horse"
X_train
typeof(X_train)
test(nn,X_train)

Vector <: AbstractArray

C_train
nc=length(unique(C_train))
n = length(C_train)
Cp=zeros(Int64, nc,n)
for (i,c) in enumerate(C_train)
    Cp[c,i] = 1
end

x = [1.0 , 2 ,3]
x = [1.0  2 3; 2.0  5.0  4]
nh=test(nn,x)


struct SummedArray{T<:Number,S<:Number}
           data::Vector{T}
           sum::S
           function SummedArray(a::Vector{T}) where T
               S = widen(T)
               new{T,S}(a, sum(S, a))
           end
       end


knn=8
x=t.x
n=length(x)             #number of data points
I=zeros(knn,n)
J=zeros(knn,n)
dd = zeros(n,knn)
di = zeros(n,knn)
i=1
x[i]
x
r = similar(x)
d = zeros(n)
for j=1:n
    r[j] = x[j] .- x[i]
    r[j] = r[j].*r[j]
    d[j] = sum(r[j])
end
idx = sortperm(d[:])
# println(idx[1:knn])
# println(d[idx[1:knn]])
# println(di[i,:])
di[i,:] = d[idx[1:knn]]
I[:,i] = i.*ones(knn,1)
J[:,i] = idx[1:knn]



n=size(A,1)
I,J,_=findnz(A)

I = [1 , 2]
J = [3 , 5]
X=t.x
aa = zeros(length(I))
for (idx,(i,j)) in enumerate(zip(I,J))
    aa[idx]=sum((X[i].-X[j]).^2)
end
aa
V = exp.(-sum((X[:,I].-X[:,J]).^2,dims=1)./ϵ)
W = SparseArrays.sparse(I,J,V[:],n,n);
Ws=sum(W,dims=2)
D = LinearAlgebra.Diagonal(Ws[:])
Dh = LinearAlgebra.Diagonal(1 ./ sqrt.(Ws[:]))
L = D-W;
L = Dh*L*Dh;
L = 0.5*(L+L') #Get rid of small asymmetries
# println("L",typeof(L))

map(sum, eachslice(t.x, dims=1))
t.x

rr = rand(50,50,50,50);
@time kk=sumdrop(rr)

sumdrop(A, d; _dims = ntuple(i -> i<d ? i : i+1, ndims(A)-1)) = dropdims(sum(A; dims=_dims); dims=_dims)
using BenchmarkTools
sumdrop(A; dims=ntuple(i->i+1, ndims(A)-1)) = dropdims(sum(A, dims=dims), dims=dims)

n=4
sum(rr,dims=setdiff([1:ndims(rr);],[n]))
sumdrop(rr,4)
rr = rand(45,46,47,50);
@benchmark map(sum, eachslice(rr, dims=4))
@benchmark sum(rr, dims=(1,2,3))
@benchmark sumdrop(rr,4)
@benchmark dropdims(sum(rr,dims=setdiff([1:ndims(rr);],[n])),setdiff([1:ndims(rr);],[n]))

typeof(setdiff([1:ndims(rr);],[n]))
sum(rr,dims=setdiff([1:ndims(rr);],[n])),setdiff([1:ndims(rr);],[n])



sumdrop(A, d) = dropdims(sum(A,dims=setdiff([1:ndims(A);],[d])),dims=(setdiff([1:ndims(A);],[d])...,))

sumdrop(rr,3)

xx=sum(rr,dims=setdiff([1:ndims(rr);],[n]))
dropdims(xx,dims=(1,2,3))

aa=(setdiff([1:ndims(rr);],[n])...,)
dropdims(xx,dims=aa)
dropdims(xx,dims=(setdiff([1:ndims(rr);],[n])...,))

dropdims(sum(rr,dims=setdiff([1:ndims(rr);],[n])),dims=(setdiff([1:ndims(rr);],[n])...,))


dropdims(sum(rr,dims=setdiff([1:ndims(rr);],[n])),(1,2,3))
ntuple(i->i+1, ndims(t.x)-1)
dropdims(rand(1,1,3), dims=(1:2))
typeof((setdiff([1:ndims(rr);],[n])...,))
typeof((1,2))
1:2

n=size(rr,4)
rr_2d=reshape(rr, :,n)

a=Conv((3,3), 3=>16)
aa=Conv((3,3), 3=>16)
aa.weight = a.weight
a.weight[1,1,1,1]
b=Chain(a,aa,a)
Conv
length(b.layers)
b.layers[1].weight
b.layers[2].weight

w = param(rand(5, 10))
function m(x)
  encoding = w*x
  decoding = w'*encoding
end




using Flux
import Base: transpose
size(d.W)
transpose(d.W)
transpose(d::Dense) = Dense(size(d.W)..., d.σ)
transpose(d::Dense) = Dense(transpose(d.W)..., d.σ)
d.W
d.W[:,1]
d
Dense((d.W[:,1],d.W[:,2]))
d.W[:,1]
function transpose(c::Conv)
    s = size(c.weight)
    size_ = s[1:end-2]
    in, out = s[end-1:end]
    ConvTranspose(size_, out=>in, c.σ, pad=c.pad, stride=c.stride, dilation=c.dilation)
end

function transpose(c::ConvTranspose)
    s = size(c.weight)
    size_ = s[1:end-2]
    in, out = s[end-1:end]
    Conv(size_, in=>out, c.σ, pad=c.pad, stride=c.stride, dilation=c.dilation)
end

d = Dense(2, 3, relu)
c = Conv((3, 3), 3=>4)
ct = ConvTranspose((3, 3), 4=>3)
d.W
dt=transpose(d) # Dense(15, 10, NNlib.relu)



dt.W
transpose(c) # ConvTranspose((3, 3), 4=>3)
transpose(ct) #


d.W[:,1]
d
Dense((d.W[:,1],d.W[:,2]))
d.


c.k
σ = tanh
d.W
aa=zeros(3,2)
Dense(3,2,σ,initW=aa)
Dense(size(d.W),initW=d.W)

a=Dense(3,2)
a.W
a.b
b=Dense(a.W,a.b)
using Flux: MeanPool
n=4
σ = relu

conv1=Conv((3,3), 16=>32)
conv2=Conv((3,3), 32=>64)
conv3=Conv((3,3), 64=>128)
forward1 = Chain(
  Conv((3,3), 3=>16),BatchNorm(16, σ),
  IdentitySkipConv((conv1,BatchNorm(32, σ),Conv((3,3), 32=>32)),dt,conv1),
  IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
  IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
  IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
  IdentitySkipConv((conv2,BatchNorm(64, σ),Conv((3,3), 64=>64)),dt,conv2),
  IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
  IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
  IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
  IdentitySkipConv((conv3,BatchNorm(128, σ),Conv((3,3), 128=>128)),dt,conv3),
  IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
  IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
  IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
  BatchNorm(128, σ),MeanPool((8,8))) |> device


  struct IdentitySkipConv
     inner
     dt
     outer
  end
  (m::IdentitySkipConv)(x) = m.dt .* m.inner(x) .+ m.outer(x)
  @Flux.treelike IdentitySkipConv
forward1 = Chain(
    Conv((3,3), 3=>16),BatchNorm(16, σ),
    IdentitySkipConv((conv1,BatchNorm(32, σ),Conv((3,3), 32=>32)),dt,Conv((1,1),16=>32)),
    IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
    IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
    IdentitySkip((Conv((3,3), 32=>32),BatchNorm(32, σ),Conv((3,3), 32=>32)),dt),
    IdentitySkipConv((conv2,BatchNorm(64, σ),Conv((3,3), 64=>64)),dt,Conv((1,1),32=>64)),
    IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
    IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
    IdentitySkip((Conv((3,3), 64=>64),BatchNorm(64, σ),Conv((3,3), 64=>64)),dt),
    IdentitySkipConv((conv3,BatchNorm(128, σ),Conv((3,3), 128=>128)),dt,Conv((1,1),64=>128)),
    IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
    IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
    IdentitySkip((Conv((3,3), 128=>128),BatchNorm(128, σ),Conv((3,3), 128=>128)),dt),
    BatchNorm(128, σ),MeanPool((8,8))) |> device



ps = params(forward1) #An object referencing all the parameters

sum(length, ps) #The total number of parameters

sum(length(p) for p ∈ ps) #The total number of parameters

length(forward.layers)
forward.layers[3].inner
length(forward.layers[1].weight)
Flux.params(forward)
classify = Chain(
  Dense(128,10),softmax) |> device

  x -> maxpool(x, PoolDims(x,(2,2))),
  Conv((5,5), 16=>8, relu),
  x -> maxpool(x, PoolDims(x,(2,2))),
  x -> reshape(x, :, size(x, 4)),
  Dense(200, 120),
  Dense(120, 84)) |> device

IdentitySkip(Conv)








a=Conv((3,3),3=>16)
b=Conv(a.weight,a.bias)
b.weight















dt = 0.1
conv1=Conv((3,3), 16=>32,pad=1)
conv2=Conv((3,3), 32=>64)
conv3=Conv((3,3), 64=>128)



inner=Chain(Conv((3,3), 16=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1))


forward = Chain(
  Conv((3,3), 3=>16,pad=1),BatchNorm(16, σ),
  IdentitySkipConv(inner,dt,Conv((1,1),16=>32)))|> device

forward1 = Chain(
  Conv((3,3), 3=>16,pad=1),BatchNorm(16, σ),
  inner)|> device

forward2 = Chain(
  Conv((3,3), 3=>16,pad=1),BatchNorm(16, σ),
  Conv((1,1),16=>32))|> device

X=t.x[:,:,:,1:2]
forward1(X)
forward2(X)
forward(X)



inner=Chain(conv1,BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1))
device=cpu
forward = Chain(
  Conv((3,3), 3=>16),BatchNorm(16, σ),
  IdentitySkipConv(inner,dt,Conv((1,1),16=>32)))|> device

forward1 = Chain(
  Conv((3,3), 3=>16,pad=1),BatchNorm(16, σ),
  inner)|> device

forward2 = Chain(
  Conv((3,3), 3=>16,pad=1),BatchNorm(16, σ),
  Conv((1,1),16=>32))|> device

forward0 = Chain(
  Conv((3,3), 3=>16,pad=1),BatchNorm(16, σ))|> device



ps = params(forward) #An object referencing all the parameters

sum(length(p) for p ∈ ps) #The total number of parameters

di = zeros(n,knn)
for i=1:n
    di[i,:] = d[i]
end
conv0=Conv((3,3), 3=>16,pad=1,stride=2)
conv1=Conv((3,3), 16=>32,pad=1,stride=2)
conv3=Conv((3,3), 64=>128,pad=1)
conv2=Conv((3,3), 32=>64,pad=1)
conv0(X)

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

poo=MeanPool((8,8))
conv1(X)
classify = Chain(
  Dense(128,10),softmax) |> device

X
Y=forward(X)
U=classify(Y)

x= [32, 3.2]
y = [5]
y=oftype(x,y)
knn=9
n=20
I=zeros(Int32,knn,n)
x=t.x
using EllipsisNotation
sumdrop(A, d) = dropdims(sum(A,dims=setdiff([1:ndims(A);],[d])),dims=(setdiff([1:ndims(A);],[d])...,)) #sums all dimensions but the designated dimension and collapse all other dimensions
println(typeof(x))
nd = ndims(x)           #Number of dimensions
n = size(x,nd)          #number of data points
I=zeros(Int32,knn,n)
J=zeros(Int32,knn,n)
di = zeros(Float32,n,knn)
for i=1:n
  r = x .- x[..,i]
  d=sumdrop(r.*r , nd)
  idx = sortperm(d[:])
  di[i,:] = d[idx[1:knn]]
  I[:,i] = i.*ones(knn,1)
  J[:,i] = idx[1:knn]
end
using SparseArrays
A = (SparseArrays.sparse(I[:],J[:],ones(Float32,length(J[:]),1)[:],n,n))
A = A+A'  # Warning this is not a normal adjencymatrix, since we have values that are 2 instead of 1 in it as well!
println(typeof(A))
return A, Statistics.median(di)

X=x
using Statistics
using LinearAlgebra
di
ϵ =Statistics.median(di)
typeof(ϵ)
n=size(A,1)
I,J,_=findnz(A)
nd = ndims(X)
aa=sumdrop((X[..,I] .- X[..,J]).^2 , nd)
V = exp.(-aa./ϵ)
W = SparseArrays.sparse(I,J,V[:],n,n)
Ws=sum(W,dims=2)
D = LinearAlgebra.Diagonal(Ws[:])
Dh = LinearAlgebra.Diagonal(1 ./ sqrt.(Ws[:]))
L = D-W
L = Dh*L*D
L = oftype(L[1],0.5)*(L+L') #Get rid of small asymmetries



using EllipsisNotation
@time y = forward(t.x[..,10:20])
y = forward[1](t.x[..,1:5])
y = forward[1:2](t.x[..,1:5])
y = forward[1:3](t.x[..,1:5])
y = forward[1:4](t.x[..,1:5])

y = forward(t.x[..,1:5])
println("y",typeof(y))
@time u = classify(y)
println("u",typeof(u))
@time misfit_j = Flux.crossentropy(u, t.cp[:,10:20])
# misfit_j = Statistics.mean(crossentropy(u, t.cp[:,batch]))
t.cgp[:,10:20] = Tracker.data(u)
@time back!(misfit_j)
t10=time()
optimizer()

function forward_w_hist(X,chain) # TODO - This should overload Chain instead
   yl=[]
   y=chain[1](X)
   push!(yl,y)
   for i in range(2,stop=length(chain))
     y=chain[i](y)
     push!(yl,y)
   end
   return yl
end

@time ylreg = forward_w_hist(t.x[..,1:10],forward)
println("ylreg",typeof(ylreg))
