module NeuralNetworks
    include("./../src/nn.jl")
    import .LoadCustomLayers.IdentitySkip
    import .LoadCustomLayers.IdentitySkipConv
    import .LoadCustomLayers.MyDense
    using Flux
    using CuArrays

    export WideNet
    function WideNet(nblocks,layers_pr_block,ch;pixels_in=32,dt=1,nc=10,device=cpu)
        layers=[]
        ResLayers = []

        push!(layers, Conv((3,3), 3=>ch[1],pad=1))
        push!(ResLayers,0)
        for i=1:nblocks
            if i == 1
                stride = 1
            else
                stride = 2
            end
            push!(layers, Chain(BatchNorm(ch[i], σ),IdentitySkipConv(Chain(Conv((3,3), ch[i]=>ch[i+1],pad=1,stride=stride),BatchNorm(ch[i+1], σ),Conv((3,3), ch[i+1]=>ch[i+1],pad=1)),dt,Conv((1,1), ch[i]=>ch[i+1],stride=stride))))
            push!(ResLayers,1)
            for j=2:layers_pr_block
                push!(layers, Chain(BatchNorm(ch[i+1], σ),IdentitySkip(Chain(Conv((3,3), ch[i+1]=>ch[i+1],pad=1),BatchNorm(ch[i+1], σ),Conv((3,3), ch[i+1]=>ch[i+1],pad=1)),dt)))
                push!(ResLayers,1)
            end
        end
        pixels_out = Int(pixels_in/(2)^(nblocks-1))
        push!(layers, Chain(BatchNorm(ch[end], σ),MeanPool((pixels_out,pixels_out)),x -> dropdims(x,dims=(1,2))))
        push!(ResLayers, 0)
        forward = Chain(layers...) |> device
        classify = Chain(Dense(ch[end],nc),softmax) |> device

        return forward,ResLayers,classify
    end
end


  # layer1 = Conv((3,3), 3=>16,pad=1)
  # layer2 = Chain(BatchNorm(16, σ),IdentitySkipConv(Chain(Conv((3,3), 16=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt,Conv((1,1),16=>32)))
  # layer3 = Chain(BatchNorm(32, σ),IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt))
  # layer4 = Chain(BatchNorm(32, σ),IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt))
  # layer5 = Chain(BatchNorm(32, σ),IdentitySkip(Chain(Conv((3,3), 32=>32,pad=1),BatchNorm(32, σ),Conv((3,3), 32=>32,pad=1)),dt))
  # layer6 = Chain(BatchNorm(32, σ),IdentitySkipConv(Chain(Conv((3,3), 32=>64,pad=1,stride=2),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt,Conv((1,1),32=>64,stride=2)))
  # layer7 = Chain(BatchNorm(64, σ),IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt))
  # layer8 = Chain(BatchNorm(64, σ),IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt))
  # layer9 = Chain(BatchNorm(64, σ),IdentitySkip(Chain(Conv((3,3), 64=>64,pad=1),BatchNorm(64, σ),Conv((3,3), 64=>64,pad=1)),dt))
  # layer10 = Chain(BatchNorm(64, σ),IdentitySkipConv(Chain(Conv((3,3), 64=>128,pad=1,stride=2),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt,Conv((1,1),64=>128,stride=2)))
  # layer11 = Chain(BatchNorm(128, σ),IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt))
  # layer12 = Chain(BatchNorm(128, σ),IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt))
  # layer13 = Chain(BatchNorm(128, σ),IdentitySkip(Chain(Conv((3,3), 128=>128,pad=1),BatchNorm(128, σ),Conv((3,3), 128=>128,pad=1)),dt))
  # layer14 = Chain(BatchNorm(128, σ),MeanPool((8,8)),x -> dropdims(x,dims=(1,2)))
