module Vizualize
 using Plots

 function Plot(X,C)
     plotly()
     # plotly(show = true)
     # gr()
     p1=plot()
     Cu=unique(C)
     for Ci in Cu
         # println(Ci)
         # check= vec(C.==Ci)
         # println(check)
         # println(size(check))
         p1=plot!(X[1,vec(C.==Ci)],X[2,vec(C.==Ci)],X[3,vec(C.==Ci)],seriestype=:scatter,leg=false)

     end
     # display(p1)
     gui(p1)
 end



end
