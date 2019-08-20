module DataGenerators

    export squares

    include("Misc.jl")
    using .Misc
    using Metalhead, Images
    using Metalhead: trainimgs
    using Metalhead: valimgs

    function squares(n)
        vec=range(-1,1,length=n)
        x=zeros(n,n)
        y=zeros(n,n)
        for i =1:n
            x[i,:]=vec
            y[:,i]=vec
        end
        X=vcat(x[:]',y[:]',0*y[:]')
        labelss = zeros(n,n)
        ii = (abs.(x) .< 0.75) .& (abs.(y) .< 0.75)
        labelss[ii] .= 1
        C = vcat(labelss[:]', 1 .- labelss[:]')
        return X,C
    end

    function cifar10(ntrain,nval,shuffle)
        Metalhead.download(CIFAR10)
        getarray(X) = float.(permutedims(channelview(X), (2, 3, 1)))
        X_train= 0
        C_train = 0
        X_val = 0
        C_val = 0
        if ntrain > 0
            train = trainimgs(CIFAR10)
            xx=collect(range(1,stop=length(train)))
            idx,_=sample_wo_repl!(xx,ntrain,shuffle)
            C_train=[train[i].ground_truth.class for i in idx]
            X = [getarray(train[i].img) for i in idx]
            X_train = Array{Float32}(undef, 32,32,3,ntrain)
            for (i,xi) in enumerate(X)
                X_train[:,:,:,i] = xi
            end

        end
        if nval > 0
            val = valimgs(CIFAR10)
            xx=collect(range(1,stop=length(val)))
            idx,_=sample_wo_repl!(xx,nval,shuffle)
            C_val=[val[i].ground_truth.class for i in idx]
            X = [getarray(val[i].img) for i in idx]
            X_val = Array{Float32}(undef, 32,32,3,nval)
            for (i,xi) in enumerate(X)
                X_val[:,:,:,i] = xi
            end
        end
        return X_train,C_train,X_val,C_val
    end

    function SelectKnownPoints(C::AbstractArray{<: Real,1};n_known::Int=-1,n_known_of_each::Int=-1,nc::Int=-1)
        if nc == -1
            nc = length(unique(C))
        end
        if n_known_of_each <= 0 & n_known <= 0
            error("Number of known points to select not set.")
        elseif n_known_of_each > 0 & n_known > 0
            error("Both n_known_of_each and n_known are set, only one should be set.")
        elseif n_known_of_each > 0
            # ik = zeros(Int64, nc*n_known_of_each)
            ik = Array{Int64,1}(undef,0)
            labels_found = zeros(nc)
            for (idx,ci) in enumerate(C)
                if labels_found[Int(ci)] < n_known_of_each
                    ik = append!(ik, idx)
                    labels_found[Int(ci)] += 1
                else
                    continue
                end
                if sum(labels_found) >= nc*n_known_of_each
                    break
                end
            end
            if sum(labels_found) < nc*n_known_of_each
                println(labels_found)
                println(ik)
                error("Not enough labels was found.")
            end
        else
            ik=collect(range(0, stop=min(n_known, length(C))))
        end
        return ik
    end

end
