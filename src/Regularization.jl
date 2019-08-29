module Regularization

    using SparseArrays
    using Statistics
    using LinearAlgebra
    using Flux.Tracker
    using EllipsisNotation

    export Laplacian

    sumdrop(A, d) = dropdims(sum(A,dims=setdiff([1:ndims(A);],[d])),dims=(setdiff([1:ndims(A);],[d])...,)) #sums all dimensions but the designated dimension and collapse all other dimensions

    function Laplacian(x::Array{<: AbstractFloat,T};track=true) where T
        # if mode == 0
        A,ϵ = AdjacencyMatrix(x,track=track)
        L = GraphLaplacian(x,A,ϵ,track=track)
        return L
    end

    function Laplacian(x,laplacelayers::Vector{<: Int} ;track=true)
        L=[] #TODO update its type
        for i=1:length(laplacelayers)
            A,ϵ = AdjacencyMatrix(x[i],track=track)
            Li = GraphLaplacian(x[i],A,ϵ,track=track)
            push!(L,Li)
        end
        return L
    end

    function Laplacian(x;track=true)
        # println(typeof(x[end]))
        A,ϵ   = AdjacencyMatrix(x[end],track=track)
        L = GraphLaplacian(x[end],A,ϵ,track=track)
        return L
    end

    function AdjacencyMatrix_(x;nn=9)
      nd = ndims(x)           #Number of dimensions
      n = size(x,nd)          #number of data points
      knn = min(n,nn)
      I=zeros(Int64,knn,n)
      J=zeros(Int64,knn,n)
      di = zeros(typeof(x[1]),n,knn)  #TODO Fix all cases like this to instead use eltype.
      for i=1:n
          r = x .- x[..,i]
          d=sumdrop(r.*r , nd)
          idx = sortperm(d[:])
          di[i,:] = d[idx[1:knn]]
          I[:,i] = i.*ones(knn,1)
          J[:,i] = idx[1:knn]
      end
      A = SparseArrays.sparse(I[:],J[:],ones(typeof(x[1]),length(J[:]),1)[:],n,n)
      A = A+A'  # Warning this is not a normal adjencymatrix, since we have values that are 2 instead of 1 in it as well!
      return A, Statistics.median(di)
    end

    function AdjacencyMatrix(x;knn=9,track)
      if track
          A, di = AdjacencyMatrix_(x,knn=knn)
          # println("A",typeof(A))
      else
          A, di = AdjacencyMatrix_(Tracker.data(x),nn=knn)
          # println("A",typeof(A))
      end
      return A, di
    end

    function SetLaplaceLayers(laplace_mode)
        if laplace_mode == 0 #Input laplacian
            laplacelayers = [0]
        elseif laplace_mode == 1 #Just before classification laplacian

        elseif laplace_mode == 2 #Interpolated laplacian

        elseif laplace_mode == 3 #Full laplacian

        end
    end


    function GraphLaplacian_(X,A,ϵ)
        n=size(A,1)
        I,J,_=findnz(A)
        nd = ndims(X)
        aa=sumdrop((X[..,I] .- X[..,J]).^2 , nd)
        V = exp.(-aa./ϵ)
        W = SparseArrays.sparse(I,J,V[:],n,n);
        Ws=sum(W,dims=2)
        D = LinearAlgebra.Diagonal(Ws[:])
        Dh = LinearAlgebra.Diagonal(1 ./ sqrt.(Ws[:]))
        L = D-W;
        L = Dh*L*Dh;
        L = oftype(L[1],0.5)*(L+L') #Get rid of small asymmetries
        # println("L",typeof(L))
        return L
    end

    function GraphLaplacian(X,A,ϵ;track=true)
        if track
            L = GraphLaplacian_(X,A,ϵ)
        else
            L = GraphLaplacian_(Tracker.data(X),A,ϵ)
        end
        return L
    end


    function Compute_Regularization(yl,L0)
        reg = 0
        n = size(L0,1)
        # println("L0",typeof(L0))
        # println("L0 size",size(L0))
        # println("L0 nnz",nnz(L0))
        for layer in yl
            # println("layer",typeof(layer))
            layer_flat=reshape(layer, :,n)
            reg += tr(layer_flat*L0*transpose(layer_flat))
        end
        return reg
    end

    function Compute_Regularization(vl,L,resnetlayers,laplacelayers)
      reg = 0
      n = size(L[1],1)
      for (i,layeridx) in enumerate(resnetlayers)
        layer = vl[i]
        layer_flat=reshape(layer, :,n)
        if i in laplacelayers
          laplaceidx = findfirst(isequal(i),laplacelayers)
          Li = L[laplaceidx]
        else
          layers1 = [t for t in laplacelayers if t < i]
          layers2 = [t for t in laplacelayers if t > i]
          if !isempty(layers1) & !isempty(layers2)
            layer1 = maximum(layers1)
            layer2 = minimum(layers2)
            nl = layer2 - layer1
            Li = L[findfirst(isequal(layer1),laplacelayers)] * (nl-(i-layer1))/nl +  L[findfirst(isequal(layer2),laplacelayers)] * (nl-(layer2-i))/nl
          elseif !isempty(layers1)
            layer = maximum(layers1)
            Li = L[findfirst(isequal(layer),laplacelayers)]
          elseif !isempty(layers2)
            layer = minimum(layers2)
            Li = L[findfirst(isequal(layer),laplacelayers)]
          else
            error("no fitting laplace layer was found, this shouldn't happen")
          end
        end
        reg += tr(layer_flat*Li*transpose(layer_flat))
      end
      return reg
    end

    function Compute_Regularization(yl,L0,Ln)
        reg = 0
        nl = length(yl)
        n = size(L0,1)
        # println("L0",typeof(L0))
        # println("Ln",typeof(Ln))
        for (i,layer) in enumerate(yl)
            Li = L0 * ((nl - i ) / (nl-1)) + Ln * ((i -1) / (nl-1))
            # println("layer",typeof(layer))
            # println("Li",typeof(Li))
            layer_flat=reshape(layer, :,n)
            reg += tr(layer_flat*Li*transpose(layer_flat))
        end
        return reg
    end

end
