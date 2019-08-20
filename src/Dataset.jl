module Data

   export maxprob

   function maxprob(cp)
      _,c = findmax(cp,dims=1)
      return getindex.(c,1)
   end

   struct tSet{T}
      name::String
      x :: AbstractArray{<: AbstractFloat, T}   #data
      cp::AbstractArray{<: AbstractFloat, 2}    #ground truth
      cgp::AbstractArray{<: AbstractFloat, 2}   #guess
      ik::Array{<: Signed,1}       #known indices
      iu::Array{<: Signed,1}       #unknown indices
      function tSet(name::String,x::AbstractArray{<: AbstractFloat, T},cp::AbstractArray{<: AbstractFloat,2},ik::Array{<: Signed,1}) where T
         cgp=zeros(size(cp))
         iu=collect(setdiff(Set(range(1, stop=size(cp,2))),Set(ik)))
         new{T}(name,x,cp,cgp,ik,iu)
      end
   end


   struct Dataset
   #   nlabels_known :: Int
      batch_size :: Int
      batch_shuffle :: Bool
      nc :: Int
      nf :: Int
      # train :: tSet
      # val :: tSet
   end

   function InitDataset(str,X,cp;ik,batch_size,batch_shuffle,device=cpu)
      nf=size(X,1)
      if ndims(cp) == 1
         nc=length(unique(cp))
         n = length(cp)
         Cp=zeros(typeof(X[1]), nc,n) |> device
         for (i,c) in enumerate(cp)
             Cp[Int(c),i] = 1
         end
      else
         Cp=cp |> device
         nc=size(cp,1)
      end

      d=Dataset(batch_size,batch_shuffle,nc,nf)
      t=tSet(str,X,Cp,ik)
      return d,t



   end

end
