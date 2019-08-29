module Optimization
    include("Misc.jl")
    include("Regularization.jl")
    include("Dataset.jl")
    import .Misc.sample_wo_repl
    using .Regularization
    using Flux
    using Flux.Tracker
    using Printf
    using Statistics
    using .Data
    using EllipsisNotation


    function forward_w_hist(X,chain,resnetlayers,laplacelayers) # TODO - This should overload Chain instead
       #We assume that both laplacelayers and resnetlayers are ordered, if not this could be dangerous
       yl=[]
       vl=[]
       println("resnetelayers",resnetlayers)
       println("laplacelayers",laplacelayers)
       if 0 in laplacelayers
           push!(yl,X) #include input
       end
       if 1 in resnetlayers
           y=chain[1][1:end-1](X)
           push!(vl,y[1]) #Only take the first component of the split, which should be F(x,θ)
           y=chain[1][end](y)
       else
           y=chain[1](X)
       end
       if 1 in laplacelayers
           push!(yl,y)
       end
       for i in range(2,stop=length(chain))
         y=chain[i][1:end-1](y)
         if i in resnetlayers
             push!(vl,y[1])
         end
         y=chain[i][end](y)
         if i in laplacelayers
             push!(yl,y) #take the whole layer
         end
       end
       return yl,vl
    end

 function training!(forward,classify,optimizer,epochs,α,trainingset,validationset,batch_size,reg_batch_size,batch_shuffle,laplace_mode,track_laplacian,laplacelayers,resnetlayers)
    t=trainingset
    v=validationset
    loss_min = Inf
    best_epoch = 0
    time_spent = 0
    val_batch_size = 100
    forward_weights = Tracker.data.(Flux.params(forward))
    classify_weights = Tracker.data.(Flux.params(classify))
    t0=time()
    println("Ite     loss      misfit      regu        acc_u      acc_k      acc_v     time(s)   cBest  α")
    for epoch in range(1,stop=epochs)
        acc_u=0.0
        acc_k=0.0
        acc_val=0.0
        misfit = 0.0
        reg = 0.0
        loss = 0.0
        cBest = ""
        if α != 0 #Compute regularization
            reg_batch, _ = Misc.sample_wo_repl!(vcat(t.ik,t.iu),reg_batch_size,batch_shuffle)
        end
        ik=copy(t.ik)
        while true #Loop through all known labels
            # println("data x, type: ",typeof(Tracker.data(t.x[1])))
            reg_j = oftype(Tracker.data(t.x[1]),0.0)
            batch, ik = Misc.sample_wo_repl!(ik,batch_size,batch_shuffle) # This will not work right now, what if batch is lar4ger than the samples
            if isempty(batch)
                break
            end
            y = forward(t.x[..,batch])
            u = classify(y)
            misfit_j = Flux.crossentropy(u, t.cp[:,batch])
            t.cgp[:,batch] = Tracker.data(u)
            if α != 0
                @time yl,vl = forward_w_hist(t.x[..,reg_batch],forward,resnetlayers,laplacelayers)
                @time L = Regularization.Laplacian(yl,track=track_laplacian,laplacelayers)
                @time reg_j = Regularization.Compute_Regularization(vl,L,resnetlayers,laplacelayers)
                # println(typeof(reg_j))
                reg_j = oftype(reg_j,α) * reg_j / oftype(reg_j,length(reg_batch))
            end
            loss_j = misfit_j + reg_j
            misfit += Tracker.data(misfit_j)
            reg += Tracker.data(reg_j)
            loss += Tracker.data(loss_j)
            # println(typeof(loss_j))
            back!(loss_j)
            optimizer()
        if loss < loss_min
            loss_min = loss
            cBest = '*'
            best_epoch = epoch
            forward_weights = Tracker.data.(Flux.params(forward))
            classify_weights = Tracker.data.(Flux.params(classify))
        end
        if ~ isempty(t.ik)
            acc_k = Statistics.mean(maxprob(t.cp[:,t.ik]) .== maxprob(t.cgp[:,t.ik]))*100
        end
        # if ~ isempty(v.x)
        #     println("validation")
        #     val_batch, _ = Misc.sample_wo_repl!(vcat(v.ik,v.iu),val_batch_size,batch_shuffle)
        #     y_val = forward(v.x[..,val_batch])
        #     u_val = classify(y_val)
        #     v.cgp[:,val_batch] = Tracker.data(u_val)
        #     acc_val = Statistics.mean(maxprob(v.cp[:,val_batch]) .== maxprob(t.cgp[:,val_batch]))*100
        # end
        time_spent=time()-t0
        @printf("%3d %10.2e %10.2e %10.2e %10.2f %10.2f %10.2f %10.2f    %1s   %2.2e \n", epoch,Tracker.data(loss),Tracker.data(misfit),Tracker.data(reg),acc_u,acc_k,acc_val,time_spent,cBest,α)
        end
    end
    #Select the best weights for the network
    Flux.loadparams!(forward, forward_weights)
    Flux.loadparams!(classify, classify_weights)
    return best_epoch, loss_min, time_spent
 end
end
