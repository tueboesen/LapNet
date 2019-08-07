module Misc

    export sample_wo_repl!

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


end


# function fwd(X,chain) # To do - This should overload Chain instead
#   yl=[]
#   y=chain[1](X)
#   push!(yl,y)
#   for i in range(2,length(chain))
#     y=chain[i](y)
#     push!(yl,y)
#   end
#   return yl
# end
#
# fwd(X,forward)

# methodswith(typeof(forward.layers); supertypes=true) #Usefull trick
