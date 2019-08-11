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


    function forward_w_hist(X,chain) # TODO - This should overload Chain instead
       yl=[]
       y=chain[1](X)
       #push!(yl,y) Commented out because we want to skip the first layer in regularization
       for i in range(2,stop=length(chain)-1) #removed last layer for wideNet28 as well, since this isn't a residual layer, and we do not want to compute laplacian based on it.
         y=chain[i](y)
         push!(yl,y)
       end
       return yl
    end

 function training!(forward,classify,optimizer,epochs,α,trainingset,validationset,batch_size,reg_batch_size,batch_shuffle,laplace_mode,track_laplacian)
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
            @time L0 = Regularization.Laplacian(t.x[..,reg_batch],track=track_laplacian)
        end
        ik=copy(t.ik)
        while true #Loop through all known labels
            println("data x, type: ",typeof(Tracker.data(t.x[1])))
            reg_j = oftype(Tracker.data(t.x[1]),0.0)
            batch, ik = Misc.sample_wo_repl!(ik,batch_size,batch_shuffle) # This will not work right now, what if batch is lar4ger than the samples
            if isempty(batch)
                break
            end
            @time y = forward(t.x[..,batch])
            @time u = classify(y)
            misfit_j = Flux.crossentropy(u, t.cp[:,batch])
            t.cgp[:,batch] = Tracker.data(u)
            if α != 0
                @time ylreg = forward_w_hist(t.x[..,reg_batch],forward)
                if laplace_mode != 0
                    @time Ln = Regularization.Laplacian(ylreg,track=track_laplacian)
                    @time reg_j = Regularization.Compute_Regularization(ylreg,L0,Ln)
                else
                    @time reg_j = Regularization.Compute_Regularization(ylreg,L0)
                end
                println(typeof(reg_j))
                reg_j = oftype(reg_j,α) * reg_j / oftype(reg_j,length(reg_batch))
            end
            loss_j = misfit_j + reg_j
            misfit += Tracker.data(misfit_j)
            reg += Tracker.data(reg_j)
            loss += Tracker.data(loss_j)
            println(typeof(loss_j))
            @time back!(loss_j)
            @time optimizer()
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
