# Copyright (c) 2018 - 2020 Chong Ma
 
# This file contains the kernel function for the R package smog. 
# The function smog is written for the generalized linear model constraint 
# on specified hierarchical structures by using overlapped group penalty. 
# It is implemented by combining the ISTA and ADMM algorithms, and works 
# for continuous, multimonial and survival data. 


#' Generalized linear model constraint on hierarchical structure
#' by using overlapped group penalty
#' 
#' \code{smog} fits a linear non-penalized phynotype (demographic) variables such as 
#' age, gender, treatment, etc, and penalized groups of prognostic effect (main effect)
#' and predictive effect (interaction effect), by satisfying the hierarchy structure:
#' if a predictive effect exists, its prognostic effect must be in the model. It can deal
#' with continuous, binomial or multinomial, and survival response variables, underlying 
#' the assumption of Gaussian, binomial (multinomial), and cox proportional hazard models,
#' respectively. It can accept \code{\link[stats]{formula}}, and output coefficients table,
#' fitted.values, and convergence information produced in the algorithm iterations.   
#' 
#' @param x a model matrix, or a data frame of dimensions n by p, 
#'          in which the columns represents the predictor variables. 
#' @param y response variable, corresponds to the family description. 
#'          When family is ``gaussian'' or ``binomial'', \code{y} ought to
#'          be a numeric vector of observations of length n; when family 
#'          is ``coxph'', \code{y} represents the survival objects, containing the 
#'          survival time and the censoring status. See \code{\link[survival]{Surv}}.
#' @param g a vector of group labels for the predictor variables.
#' @param v a vector of binary values, represents whether or not the 
#'          predictor variables are penalized. Note that 1 indicates 
#'          penalization and 0 for not penalization.
#' @param lambda a numeric vector of three penalty parameters corresponding to L2 norm,
#'               squared L2 norm, and L1 norm, respectively. 
#' @param hierarchy a factor value in levels 0, 1, 2, which represent different
#'                  hierarchical structure within groups, respectively. 
#'                  When \code{hierarchy=0}, \eqn{\lambda_2} and \eqn{\lambda_3} are 
#'                  forced to be zeroes; when \code{hierarchy=1}, \eqn{\lambda_2} is 
#'                  forced to be zero; when \code{hierarchy=2}, there is no constraint 
#'                  on \eqn{\lambda}'s. See more explainations under ``Details''. 
#' @param family a description of the distribution family for the response 
#'               variable variable. For continuous response variable,
#'               family is ``gaussian''; for multinomial or binary response
#'               variable, family is ``binomial''; for survival response
#'               variable, family is ``coxph'', respectively.
#' @param subset an optional vector specifying a subset of observations to be 
#'               used in the model fitting. Default is \code{NULL}.
#' @param rho   the penalty parameter used in the alternating direction method 
#'              of multipliers (ADMM) algorithm. Default is 1e-3.
#' @param scale whether or not scale the design matrix. Default is \code{FALSE}.
#' @param eabs  the absolute tolerance used in the ADMM algorithm. Default is 1e-3.
#' @param erel  the reletive tolerance used in the ADMM algorithm. Default is 1e-3.
#' @param LL    initial value for the Lipschitz continuous constant for 
#'              approximation to the objective function in the Majorization-
#'              Minimization (MM) (or iterative shrinkage-thresholding algorithm 
#'              (ISTA)). Default is 1.
#' @param eta   gradient stepsize for the backtrack line search for the Lipschitz
#'              continuous constant. Default is 1.25. 
#' @param maxitr the maximum iterations for convergence in the ADMM algorithm. 
#'               Default is 500.
#' @param formula an object of class ``formula'': a symbolic description of the
#'                model to be fitted. Should not include the intercept. 
#' @param data    an optional data frame, containing the variables in the model. 
#' @param ...   other relevant arguments that can be supplied to smog.
#' 
#' @return \code{smog} returns an object of class inhering from ``smog''. The 
#'         generic accessor functions \code{coef}, \code{coefficients}, 
#'         \code{fitted.value}, and \code{predict} can be used to extract
#'         various useful features of the value returned by \code{smog}.
#'         
#'         An object of ``smog'' is a list containing at least the following 
#'         components: 
#'         
#'         \item{coefficients}{a data frame containing the nonzero predictor
#'                             variables' indexes, names, and estimates. When
#'                             family is ``binomial'', the estimates have K-1 
#'                             columns, each column representing the weights for the 
#'                             corresponding group. The last group behaves the
#'                             ``pivot''.}
#'         \item{fitted.values}{the fitted mean values for the response variable,
#'                              for family is ``gaussian''. When family is 
#'                              ``binomial", the fitted.values are the probabilies
#'                              for each class; when family is ``coxph'', 
#'                              the fitted.values are survival probabilities.}
#'         \item{loglike}{the penalized log-likelihood values for each 
#'                        iteration in the algorithm.}
#'         \item{PrimalError}{the averged norms \eqn{||\beta-Z||/\sqrt{p}} for each iteration,
#'                            in the ADMM algorithm.}
#'         \item{DualError}{the averaged norms \eqn{||Z^{t+1}-Z^{t}||/\sqrt{p}} for 
#'                          each iteration, in the ADMM algorithm.}
#'         \item{converge}{the number of iterations processed in the ADMM algorithm.}
#'         \item{call}{the matched call.}
#'         \item{formula}{the formula supplied.}
#'    
#' 
#' @details The formula has the form \code{response ~ 0 + terms} where \code{terms} is
#'          a series of predictor variables to be fitted for \code{response}. For \code{gaussian} 
#'          family, the response is a continuous vector. For \code{binomial} family, 
#'          the response is a factor vector, in which the last level denotes the ``pivot''.
#'          For \code{coxph} family, the response is a \code{\link[survival]{Surv}} 
#'          object, containing the survival time and censoring status.
#'          
#'          The terms contains the non-penalized predictor variables, and many groups 
#'          of prognostic and predictive terms, where in each group the prognostic 
#'          term comes first, followed by the predictive term.
#'          
#'          The \code{hierarchy} denotes different hierachical structures within groups
#'          by adjusting the penalty parameters in the penalty function:          
#'          \deqn{\Omega(\bm{\beta}) = \lambda_1||\bm{\beta}|| +
#'          \lambda_2||\bm{\beta}||^2+\lambda_3|\beta_2|}
#'          Where \eqn{\bm{\beta}=(\beta_1,\beta_2)}. Note that \eqn{\beta_1} denotes
#'          the prognostic effect (main effect), and \eqn{\beta_2} for the predictive effect 
#'          (interactive effect), respectively. When \code{hierachy}=0, \eqn{\lambda_2}
#'          and \eqn{\lambda_3} are forced to be zero, indicating no structure within groups.
#'          When \code{hierachy}=1, \eqn{\lambda_2} is forced to be zero; and for 
#'          \code{hierachy} = 2, there is no constraints on \eqn{\lambda}'s. For \code{hierarchy}
#'          is either 1 or 2, they both admits the existence of the structure within groups.
#'          
#'          \code{rho,eabs,erel,LL,eta} are the corresponding parameters used in the itervative
#'          shrinkage-thresholding algorithm (ISTA) and the alternating direction method of 
#'          multipliers algorithm (ADMM). 
#'          
#'          Note that the missing values in the data are supposed to be dealt with in the 
#'          data preprocessing, before applying the method. 
#'          
#' @examples  
#' # require(coxed)
#' 
#' n=50;p=1000
#' set.seed(2018)
#' # generate design matrix x
#' s=10
#' x=matrix(0,n,1+2*p)
#' x[,1]=sample(c(0,1),n,replace = TRUE)
#' x[,seq(2,1+2*p,2)]=matrix(rnorm(n*p),n,p)
#' x[,seq(3,1+2*p,2)]=x[,seq(2,1+2*p,2)]*x[,1]
#' 
#' g=c(p+1,rep(1:p,rep(2,p)))  # groups 
#' v=c(0,rep(1,2*p))           # penalization status
#' 
#' # generate beta
#' beta=c(rnorm(13,0,2),rep(0,ncol(x)-13))
#' beta[c(2,4,7,9)]=0
#' 
#' # generate y
#' data1=x%*%beta
#' noise1=rnorm(n)
#' snr1=as.numeric(sqrt(var(data1)/(s*var(noise1))))
#' y1=data1+snr1*noise1
#' \dontrun{
#' lfit1=smog(x,y=y1,g,v,hierarchy=1,lambda=c(0.02,0,0.001),family = "gaussian",scale=TRUE)
#' }
#' 
#' 
#' ## generate binomial data
#' prob=exp(as.matrix(x)%*%as.matrix(beta))/(1+exp(as.matrix(x)%*%as.matrix(beta)))
#' y2=ifelse(prob>0.5,0,1)
#' \dontrun{
#' lfit2=smog(x,y=y2,g,v,hierarchy=1,lambda=c(0.025,0,0.001),family = "binomial")
#' }
#' 
#' 
#' ## generate survival data
#' \dontrun{
#' data3=sim.survdata(N=n,T=100,X=x,beta=beta)
#' y3=data3$data[,c("y","failed")]
#' y3$failed=ifelse(y3$failed,1,0)
#' colnames(y3)=c("time","status")
#' lfit3=smog(x,y=y3,g,v,hierarchy = 1,lambda = c(0.075,0,0.001),family = "coxph",LL=10)
#' }
#' 
#' 
#' @export
smog.default <- function(x, y, g, v, lambda, hierarchy, family = "gaussian", subset = NULL, rho = 1e-3, 
                         scale = FALSE, eabs = 1e-3, erel = 1e-3, LL = 1, eta = 1.25, maxitr = 500, ...){
  if(!is.null(subset)){
    x <- as.matrix(as.matrix(x)[subset,])
    y <- as.matrix(as.matrix(y)[subset,])
  }else{
    x <- as.matrix(x)
    y <- as.matrix(y)
  }
  
  g <- as.numeric(as.factor(g))
  est <- glog(y,x,g,v,lambda,hierarchy,family,rho,
              scale,eabs,erel,LL,eta,maxitr)
  
  if(family == "gaussian"){
    wx <- cbind(rep(1,nrow(x)),x)
    est$fitted.value = as.vector(wx[,est$coefficients$Id+1]%*%est$coefficients$Estimate)
    est$residuals = as.vector(y - est$fitted.value)
    
    if(!is.null(colnames(x))){
      est$coefficients$Beta = c("Intercept",colnames(x))[est$coefficients$Id+1]
      est$coefficients = est$coefficients[,c(1,3,2)]
    }
  }
  
  if(family == "binomial"){
    probTAB = as.matrix(exp(x[,est$coefficients$Id]%*%est$coefficients$Estimate)/
                       (1+exp(x[,est$coefficients$Id]%*%est$coefficients$Estimate)))
    probTAB = cbind(probTAB, 1-rowSums(probTAB))
    predClass = apply(probTAB,1,which.max)
    predProb = apply(probTAB,1,max)
    
    est$levels = sort(unique(y))
    est$fitted.value = data.frame(Class = est$levels[predClass],
                                  Prob = predProb)
    
    if(!is.null(colnames(x))){
      est$coefficients$Beta = colnames(x)[est$coefficients$Id]
      est$coefficients = est$coefficients[,c(1,ncol(probTAB)+1,2:ncol(probTAB))]
    }
  }
  
  if(family == "coxph"){
    est$fitted.value = as.vector(round(exp(-exp(x[,est$coefficients$Id]%*%est$coefficients$Estimate)),2))
    
    if(!is.null(colnames(x))){
      est$coefficients$Beta = colnames(x)[est$coefficients$Id]
      est$coefficients = est$coefficients[,c(1,3,2)]
    }
  }
  
  est$call <- match.call()
  class(est) <- "smog"
  est
}

#' @rdname smog.default
#' 
#' @seealso \code{\link{cv.smog}}, \code{\link{predict.smog}}, \code{\link{plot.smog}}.
#' @author Chong Ma, \email{chong.ma@@yale.edu}
#'  
#' @export
smog.formula <- function(formula, data=list(), g, v, lambda, hierarchy, ...){
  mf <- model.frame(formula = formula, data = data)
  x <- model.matrix(attr(mf,"terms"),data = mf)
  y <- model.response(mf)
  
  est <- smog.default(x,y,g,v,lambda,hierarchy,...)
  est$call <- match.call()
  est$formula <- formula
  est
}


#' Cross-valiation for smog 
#' 
#' \code{cv.smog} conducts the \code{nfolds} cross-validations for the whole data,
#' where one fold of the observations are used for model-testing, and the remaining 
#' data are used for model-building. It allows the \code{nfolds} to be processed
#' in parallel, in order to speed up the cross-validation. However, it can only do 
#' the cross-validations for one user-specified \code{lambda}, because \code{lambda} 
#' is a three-dimensional vector, the optimal search for \code{lambda} is quite 
#' computationally expensive. The \code{cv.smog} outputs the Akaike's Information Criterion
#' (AIC) for each testing.  
#' 
#' @inheritParams smog.default
#' @param nfolds number of folds. One fold of the observations in the data are used
#'               as the testing, and the remaining are fitted for model training. 
#'               Default is 10.
#' @param parallel Whether or not process the \code{nfolds} cross-validations in
#'                 parallel. If \code{TRUE}, use \code{\link[foreach]{foreach}} to do each 
#'                 cross-validation in parallel. Default is \code{FALSE}.
#' @param ncores number of cpu's for parallel computing. See
#'               \code{\link[parallel]{makeCluster}} and \code{\link[doParallel]{registerDoParallel}}.
#'               Default is \code{NULL}. 
#' @param ... other arguments that can be supplied to \code{smog}.
#'
#' @return the average of Akaike's Information Criterions (AICs) from
#'         the \code{nfolds} cross-validations. Note that AIC is 
#'         -2log-likelihood+2\eqn{p_l}, where \eqn{p_l} is the number 
#'         of non-zero predictor variables.
#' 
#' @details The function runs \code{smog} \code{nfolds} times. Evenly split the whole
#'          data into \code{nfolds}, and one fold of the observations are used as 
#'          the testing data, and the remaining are used for model training. After
#'          calculating the AIC for each fold of testing data, return the average of the 
#'          AICs. Note that this method does NOT search for the optimal penalty parameters 
#'          \code{lambda}, and a specific \code{lambda} should be supplied. 
#'          
#' @examples 
#' #require(plotly)
#' 
#' # generate design matrix x
#' set.seed(2018)
#' n=50;p=1000
#' s=10
#' x=matrix(0,n,1+2*p)
#' x[,1]=sample(c(0,1),n,replace = TRUE)
#' x[,seq(2,1+2*p,2)]=matrix(rnorm(n*p),n,p)
#' x[,seq(3,1+2*p,2)]=x[,seq(2,1+2*p,2)]*x[,1]
#' 
#' g=c(p+1,rep(1:p,rep(2,p)))  # groups 
#' v=c(0,rep(1,2*p))           # penalization status
#' 
#' # generate beta
#' beta=c(rnorm(13,0,2),rep(0,ncol(x)-13))
#' beta[c(2,4,7,9)]=0
#' 
#' # generate y
#' data=x%*%beta
#' noise=rnorm(n)
#' snr=as.numeric(sqrt(var(data)/(s*var(noise))))
#' y=data+snr*noise
#' 
#' l1=l2=10^(-seq(1,3,0.2))
#' cvmod=matrix(0,length(l1),length(l2))
#' \dontrun{
#' for(i in 1:length(l1)){
#'   for(j in i:length(l2)){
#'     cvmod[i,j] = cv.smog(x,y,g,v,lambda=c(l1[i],0,l2[j]),
#'                          hierarchy=1,family="gassian",nfolds=10,scale=TRUE)
#'   }
#' }
#' 
#' plot_ly(x=l1,y=l2,z=t(cvmod),type="contour",
#'         contours=list(showlabels=TRUE))%>%
#'         colorbar(title="aic")%>%
#'         layout(xaxis=list(title="lambda1"),
#'                yaxis=list(title="lambda2"))%>%
#'         config(mathjax='cdn')
#' 
#' }
#' 
#' @seealso \code{\link{smog.default}}, \code{\link{smog.formula}}, \code{\link{predict.smog}}, \code{\link{plot.smog}}.
#' @author Chong Ma, \email{chong.ma@@yale.edu}.
#' 
#' @export
cv.smog <- function(x, y, g, v, lambda, hierarchy, family = "gaussian", nfolds = 10, parallel = FALSE, ncores = NULL, ...){
  
  if(family == "gaussian"){
    wx = cbind(rep(1,nrow(x)),x)
  }
  
  if(family == "coxph"){
    model <- smog(x,y,g,v,lambda,hierarchy,family,...)
    loglike = -sum(exp(x[,coef(model)$Id]%*%coef(model)$Estimate))
  }
  
  setlist = as.integer(seq(0,nrow(x),length.out = nfolds+1))
  ncores = ifelse(parallel,ncores,1)
  
  i = 1
  if(parallel){
    `%mypar%` =  `%dopar%`
    ncores = ifelse(is.null(ncores),1,ncores)
    cl <- parallel::makeCluster(ncores)
    doParallel::registerDoParallel(cl)
  }else{
    `%mypar%` = `%do%`
  }
  
  cv_aic <- foreach::foreach(i=1:nfolds,.combine = c,
                             .packages = c("foreach","smog"))%mypar%{
    tset <- (setlist[i]+1):setlist[i+1]
    lset <- c(1:nrow(x))[-tset]
    cvmodel <- smog::smog(x,y,g,v,lambda,hierarchy,family,subset=lset,...)
    
    if(family == "gaussian"){
      tx = wx[tset,coef(cvmodel)$Id+1]
      ty <- as.vector(tx %*% coef(cvmodel)$Estimate)
      aic <- sum((y[tset] - ty)^2)+2*nrow(coef(cvmodel))
    }
    
    if(family == "binomial"){
      probTAB = as.matrix(exp(x[tset,coef(cvmodel)$Id]%*%coef(cvmodel)$Estimate)/
                            (1+exp(x[tset,coef(cvmodel)$Id]%*%coef(cvmodel)$Estimate)))
      probTAB = cbind(probTAB, 1-rowSums(probTAB))
      ty = match(y[tset],cvmodel$levels)
      aic = -2*sum(log(apply(cbind(ty,probTAB),1,function(t) t[-1][t[1]])))+2*nrow(coef(cvmodel))
    }
    
    if(family == "coxph"){
      aic = -2*(sum(exp(x[tset,coef(cvmodel)$Id]%*%coef(cvmodel)$Estimate)) + loglike) +
        2*nrow(coef(cvmodel))
    }
    
    c(aic)
  }
  
  if(parallel)  parallel::stopCluster(cl)

  return(mean(cv_aic))
}


#' predict method for objects of the class smog
#' 
#' \code{predict.smog} can produce the prediction for user-given new data, based on the
#' provided fitted model (\code{object}) in the S3method of \code{smog}. If the \code{newdata} omitted,
#' it would output the prediction for the fitted model itself. The yielded result should
#' match with the family in the provided model. See \code{\link{smog}}.
#' 
#' @param object a fitted object of class inheriting from smog.
#' @param newdata a data frame containing the predictor variables, which are
#'                used to predict. If omitted, the fitted linear predictors 
#'                are used. 
#' @param family  a description of distribution family for which the response 
#'                variable is to be predicted.  
#' @param ... additional arguments affecting the predictions produced.
#' 
#' @details If \code{newdata = NULL}, the fitted.value based on the \code{object}
#'          is used for the prediction. 
#' 
#' @return If \code{family} = ``gaussian'', a vector of prediction for the response is returned.
#'         For \code{family} = ``coxph'', a vector of predicted survival 
#'         probability is returned. When \code{family} = ``binomial'', it outputs a data
#'         frame containing the predicted group labels and the corresponding 
#'         probabilies. 
#' 
#' @seealso \code{\link{smog.default}}, \code{\link{smog.formula}}, \code{\link{cv.smog}}, \code{\link{plot.smog}}.
#' @author Chong Ma, \email{chong.ma@@yale.edu}.
#' 
#' @export
predict.smog <- function(object, newdata = NULL, family = "gaussian",...){
  if(is.null(newdata)){
    y <- fitted(object)
  }else{
    if(!is.null(object$formula)){
      x = model.matrix(object$formula, newdata)
    }else{
      x = newdata
    }
    
    if(family == "gaussian"){
      wx = cbind(rep(1,nrow(x)),x)
      y <- as.vector(wx[,coef(object)$Id+1] %*% coef(object)$Estimate)
    }
    
    if(family == "binomial"){
      probTAB = as.matrix(exp(x[,coef(object)$Id]%*%coef(object)$Estimate)/
                            (1+exp(x[,coef(object)$Id]%*%coef(object)$Estimate)))
      probTAB = cbind(probTAB, 1-rowSums(probTAB))
      predClass = apply(probTAB,1,which.max)
      predProb = apply(probTAB,1,max)
      
      y = data.frame(Class = object$levels[predClass],
                     Prob = predProb)
    }
    
    if(family == "coxph"){
      y = round(exp(-exp(x[,coef(object)$Id]%*%coef(object)$Estimate)),2)
    }
    
  }
  return(y);
}


#' plot method for objects of the class smog
#' 
#' \code{plot.smog} can produce a panel of plots for the primal errors, dual errors, 
#' and the penalized log-likelihood values, based on the provided fitted model 
#' (\code{x}) in the S3method of \code{smog}. 
#' 
#' @param x a fitted object of class inheriting from smog.
#' @param type,xlab default line types and x axis labels for the panel of plots.
#' @param caption a list of y axes labels for the panel of plots. 
#' @param ... additional arguments that could be supplied to \code{\link[graphics]{plot}}
#'            and \code{\link[graphics]{par}}.
#' 
#' @details For the panel of three plots, the \code{xlab} is ``iterations'' and the
#'          \code{type} is ``l'', by default. The \code{ylab} are ``primal error'',
#'          ``dual error'',``log-likelihood'', respectively. This panel of plots can
#'          reflect the convergence performance for the algorithm used in \code{\link{smog}}.
#' 
#' @seealso \code{\link[graphics]{par}}, \code{\link[graphics]{plot.default}}, 
#'          \code{\link{smog.default}}, \code{\link{smog.formula}}, \code{\link{cv.smog}}.
#'          
#' @author Chong Ma, \email{chong.ma@@yale.edu}.
#' 
#' @export
plot.smog <- function(x,type = "l",xlab="iteration",caption=list("primal error","dual error","log-likelihood"),...){
  op<-graphics::par(mfrow=c(2,2),...)
  graphics::plot(x$PrimalError, type=type, xlab = xlab, ylab = caption[[1]],...)
  graphics::plot(x$DualError, type=type, xlab = xlab, ylab = caption[[2]],...)
  graphics::plot(x$loglike, type=type, xlab = xlab, ylab = caption[[3]],...)
  par(op)
}


#' smog generic
#'
#' @param x an object of class from``smog''.
#' @param ... further arguments passed to or from other methods.
#' @keywords internal
#' 
#' @return the cofficients table of the object x of class from ``smog''.
#'         See \code{\link[base]{print}}.
smog <- function(x, ...) UseMethod("smog")

#' @rdname smog
#' @keywords internal
#' 
print.smog <- function(x, ...){
  cat("Call:\n")
  print(x$call)
  
  cat("\n Coefficients:\n")
  print(x$coefficients,row.names=FALSE)
}








