# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Generalized linear model constraint on hierarchical structure
#' by using overlapped group penalty
#'   
#' @param y response variable, in the format of matrix. When family is 
#'          ``gaussian'' or ``binomial'', \code{y} ought to
#'          be a matrix of n by 1 for the observations; when family
#'          is ``coxph'', y represents the survival objects, that is,
#'          a matrix of n by 2, containing the survival time and the censoring status.
#'          See \code{\link[survival]{Surv}}.
#' @param x a model matrix of dimensions n by p,in which the columns represents 
#'          the predictor variables.
#' @param g a numeric vector of group labels for the predictor variables.
#' @param v a numeric vector of binary values, represents whether or not the
#'          predictor variables are penalized. Note that 1 indicates
#'          penalization and 0 for not penalization.
#' @param lambda a numeric vector of three penalty parameters corresponding to L2 norm,
#'               squared L2 norm, and L1 norm, respectively.
#' @param hierarchy a factor value in levels 0, 1, 2, which represent different
#'                  hierarchical structure within groups, respectively. 
#'                  When \code{hierarchy=0}, \eqn{\lambda_2} and \eqn{\lambda_3} are 
#'                  forced to be zeroes; when \code{hierarchy=1}, \eqn{\lambda_2} is 
#'                  forced to be zero; when \code{hierarchy=2}, there is no constraint 
#'                  on \eqn{\lambda}'s. See \code{\link{smog}}.
#' @param family a description of the distribution family for the response
#'               variable variable. For continuous response variable,
#'               family is ``gaussian''; for multinomial or binary response
#'               variable, family is ``binomial''; for survival response
#'               variable, family is ``coxph'', respectively.
#' @param subset an optional vector specifying a subset of observations to be
#'               used in the model fitting. Default is \code{NULL}.
#' @param rho the penalty parameter used in the alternating direction method
#'            of multipliers algorithm (ADMM). Default is 1e-3.
#' @param scale whether or not scale the design matrix. Default is \code{false}.
#' @param eabs  the absolute tolerance used in the ADMM algorithm. Default is 1e-3.
#' @param erel  the reletive tolerance used in the ADMM algorithm. Default is 1e-3.
#' @param LL    initial value for the Lipschitz continuous constant for
#'              approximation to the objective function in the Majorization-
#'              Minimization (MM) (or iterative shrinkage-thresholding algorithm
#'              (ISTA)). Default is 1.
#' @param eta gradient stepsize for the backtrack line search for the Lipschitz
#'            continuous constant. Default is 1.25.
#' @param maxitr the maximum iterations for convergence in the ADMM algorithm.
#'               Default is 500.
#' @keywords internal
#' 
glog <- function(y, x, g, v, lambda, hierarchy, family = "gaussian", rho = 1e-3, scale = FALSE, eabs = 1e-3, erel = 1e-3, LL = 1, eta = 1.25, maxitr = 500L) {
    .Call('_smog_glog', PACKAGE = 'smog', y, x, g, v, lambda, hierarchy, family, rho, scale, eabs, erel, LL, eta, maxitr)
}

#' proximal operator on L1 penalty
#' 
#' @param x numeric value.
#' @param lambda numeric value for the L1 penalty parameter.
#' @keywords internal
#' 
proxL1 <- function(x, lambda) {
    .Call('_smog_proxL1', PACKAGE = 'smog', x, lambda)
}

#' proximal operator on L2 penalty
#' 
#' @param x A numeric vector.
#' @param lambda numeric value for the L2 penalty parameter.
#' @keywords internal
#' 
proxL2 <- function(x, lambda) {
    .Call('_smog_proxL2', PACKAGE = 'smog', x, lambda)
}

#' proximal operator on the composite L2, L2-Square, and L1 penalties
#' 
#' @param x A numeric vector of two.
#' @param lambda a vector of three penalty parameters. \eqn{\lambda_1} and 
#'        \eqn{\lambda_2} are L2 and L2-Square (ridge) penalties for \eqn{x} in 
#'        a group level, and \eqn{\lambda_3} is the L1 penalty for \eqn{x_2}, respectively.
#' @param hierarchy a factor value in levels 0, 1, 2, which represent different
#'                  hierarchical structure in x, respectively. When \code{hierarchy=0},
#'                  \eqn{\lambda_2} and \eqn{\lambda_3} are forced to be zeroes; when
#'                  \code{hierarchy=1}, \eqn{\lambda_2} is forced to be zero; when 
#'                  \code{hierarchy=2}, there is no constraint on \eqn{\lambda}'s. 
#'                  See \code{\link{smog}}.
#' @param d indices for overlapped variables in x.   
#' 
prox <- function(x, lambda, hierarchy, d) {
    .Call('_smog_prox', PACKAGE = 'smog', x, lambda, hierarchy, d)
}

#' Penalty function on the composite L2, L2-Square, and L1 penalties
#' 
#' @param x A vector of two numeric values, in which \eqn{x_1} represents
#'          the prognostic effect, and \eqn{x_2} for the predictive effect, 
#'          respectively. 
#' @param lambda a vector of three penalty parameters. \eqn{\lambda_1} and 
#'        \eqn{\lambda_2} are L2 and L2-Square (ridge) penalties for \eqn{x} in 
#'        a group level, and \eqn{\lambda_3} is the L1 penalty for \eqn{x_2}, respectively.
#' @param hierarchy a factor value in levels 0, 1, 2, which represent different
#'                  hierarchical structure in x, respectively. When \code{hierarchy=0},
#'                  \eqn{\lambda_2} and \eqn{\lambda_3} are forced to be zeroes; when
#'                  \code{hierarchy=1}, \eqn{\lambda_2} is forced to be zero; when 
#'                  \code{hierarchy=2}, there is no constraint on \eqn{\lambda}'s. 
#'                  See \code{\link{smog}}.
#' @param d indices for overlapped variables in x. 
#' 
penalty <- function(x, lambda, hierarchy, d) {
    .Call('_smog_penalty', PACKAGE = 'smog', x, lambda, hierarchy, d)
}

# Register entry points for exported C++ functions
methods::setLoadAction(function(ns) {
    .Call('_smog_RcppExport_registerCCallable', PACKAGE = 'smog')
})
