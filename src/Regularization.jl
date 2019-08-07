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

    function Laplacian(x;track=true)
        # println(typeof(x[end]))
        A,ϵ   = AdjacencyMatrix(x[end],track=track)
        L = GraphLaplacian(x[end],A,ϵ,track=track)
        return L
    end

    function AdjacencyMatrix_(x;knn=9)
      println(typeof(x))
      nd = ndims(x)           #Number of dimensions
      n = size(x,nd)          #number of data points
      I=zeros(Int64,knn,n)
      J=zeros(Int64,knn,n)
      di = zeros(Float64,n,knn)
      for i=1:n
          r = x .- x[..,i]
          d=sumdrop(r.*r , nd)
          idx = sortperm(d[:])
          di[i,:] = d[idx[1:knn]]
          I[:,i] = i.*ones(knn,1)
          J[:,i] = idx[1:knn]
      end
      A = SparseArrays.sparse(I[:],J[:],ones(Float64,length(J[:]),1)[:],n,n)
      A = A+A'  # Warning this is not a normal adjencymatrix, since we have values that are 2 instead of 1 in it as well!
      return A, Statistics.median(di)
    end

    function AdjacencyMatrix(x;knn=9,track)
      if track
          A, di = AdjacencyMatrix_(x,knn=knn)
          # println("A",typeof(A))
      else
          A, di = AdjacencyMatrix_(Tracker.data(x),knn=knn)
          # println("A",typeof(A))
      end
      return A, di
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
        for layer in yl
            layer_flat=reshape(layer, :,n)
            reg += tr(layer*L0*transpose(layer))
        end
        return reg
    end

    function Compute_Regularization(yl,L0,Ln)
        reg = 0
        nl = length(yl)
        n = size(L0,1)
        for (i,layer) in enumerate(yl)
            Li = L0 * ((nl - i ) / (nl-1)) + Ln * ((i -1) / (nl-1))
            layer_flat=reshape(layer, :,n)
            reg += tr(layer_flat*Li*transpose(layer_flat))
        end
        return reg
    end

end
