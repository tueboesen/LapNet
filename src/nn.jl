module LoadCustomLayers
  using Flux
  export MyDense, IdentitySkip, IdentitySkipConv
  struct MyDense{F,S,T}
    W::S
    b::T
    σ::F
  end
  MyDense(W, b) = MyDense(W, b, identity)
  function MyDense(in::Integer, out::Integer, σ = identity;
                 initW = randn, initb = zeros)
    return MyDense(param(initW(out, in)), param(initb(out)), σ)
  end
  @Flux.treelike MyDense

  function (a::MyDense)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    σ.(W*x) .+ b
  end

  struct IdentitySkip
     inner
     dt
  end
  (m::IdentitySkip)(x) = oftype(Tracker.data(x[1]),m.dt) .* m.inner(x) .+ x
  @Flux.treelike IdentitySkip

  struct IdentitySkipConv
     inner
     dt
     outer
  end
  function (m::IdentitySkipConv)(x::AbstractArray)
      dt,inner,outer = m.dt, m.inner, m.outer
      # println(typeof(Tracker.data(x)))
      # println(typeof(inner(x)))
      # println(typeof(outer(x)))
      # println(typeof(dt*x))
      # println(typeof(dt))
      # println(typeof(Tracker.data(x[1])))
      # println(typeof(oftype(Tracker.data(x[1]),dt)))
      # println(typeof(convert(typeof(Tracker.data(x[1])), dt)))
      # println(typeof(oftype(dt,Tracker.data(x[1])) .* inner(x)))
      return oftype(Tracker.data(x[1]),dt) .* inner(x) .+ outer(x)
  end
  # (m::IdentitySkipConv)(x) = m.dt .* m.inner(x) .+ m.outer(x)
  @Flux.treelike IdentitySkipConv

  function (a::Chain)(x::Any,mode)
      #mode does nothing yet, that might come later, for now it is just there to distinguish the input for chain.
      yl=[]
      y=a[1](x)
      push!(yl,y)
      for i in range(2,stop=length(a))
          y=a[i](y)
          push!(yl,y)
      end
      return yl
  end
  #     W, b, σ = a.W, a.b, a.σ
  #   σ.(W*x) .+ b
  # end


  # function Chain(x::Any,mode)
  #   #mode does nothing yet, that might come later, for now it is just there to distinguish the input for chain.
  #   yl=[]
  #   y=Chain[1](x)
  #   push!(yl,y)
  #   for i in range(2,stop=length(Chain))
  #    y=Chain[i](y)
  #    push!(yl,y)
  #   end
  #   return yl
  # end
end
