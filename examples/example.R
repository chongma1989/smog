#==========================================================#
# R examples: structural modeling by using
#             overlapped group penalty for two data sets
# Copyright (c) 2018-2020 Chong Ma
#==========================================================#

library(smog)
require(coxed)

n=50;p=1000
set.seed(2018)
# generate design matrix x
s=10
x=matrix(0,n,1+2*p)
x[,1]=sample(c(0,1),n,replace = T)
x[,seq(2,1+2*p,2)]=matrix(rnorm(n*p),n,p)
x[,seq(3,1+2*p,2)]=x[,seq(2,1+2*p,2)]*x[,1]

g=c(p+1,rep(1:p,rep(2,p)))  # groups
v=c(0,rep(1,2*p))           # penalization status

# generate beta
beta=c(rnorm(13,0,2),rep(0,ncol(x)-13))
beta[c(2,4,7,9)]=0

# generate y
data1=x%*%beta
noise1=rnorm(n)
snr1=as.numeric(sqrt(var(data1)/(s*var(noise1))))
y1=data1+snr1*noise1
lfit1=smog(x,y=y1,g,v,hierarchy=1,lambda=c(0.02,0,0.001),
           family = "gaussian",scale = TRUE)

## generate binomial data
prob=exp(as.matrix(x)%*%as.matrix(beta))/(1+exp(as.matrix(x)%*%as.matrix(beta)))
y2=ifelse(prob>0.5,0,1)
lfit2=smog(x,y=y2,g,v,hierarchy=1,lambda=c(0.04,0,0.01),
           family = "binomial")

## generate survival data
data3=sim.survdata(N=n,T=100,X=x,beta=beta)
y3=data3$data[,c("y","failed")]
y3$failed=ifelse(y3$failed,1,0)
colnames(y3)=c("time","status")
lfit3=smog(x,y=y3,g,v,hierarchy=1,lambda=c(0.075,0,0.01),
           family = "coxph",LL=10)


## cross-validation for search optimal lambda
#require(plotly)
l1=l2=10^(-seq(1,3,0.2))
cvmod=matrix(0,length(l1),length(l2))

for(i in 1:length(l1)){
  for(j in i:length(l2)){
    cvmod[i,j] = cv.smog(x,y,g,v,lambda=c(l1[i],0,l2[j]),
                         hierarchy=1,family="gassian",nfolds=10)
  }
}

plot_ly(x=l1,y=l2,z=t(cvmod),type="contour",
        contours=list(showlabels=TRUE))%>%
        colorbar(title="aic")%>%
        layout(xaxis=list(title="lambda1"),
               yaxis=list(title="lambda2"))%>%
        config(mathjax='cdn')

