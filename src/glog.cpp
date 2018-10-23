// Copyright (c) 2018 - 2020 Chong Ma
// 
// This file contains the kernel function for the R package glog. 
// The function glog is written for the generalized linear model constraint 
// on specified hierarchical structures by using overlapped group penalty. 
// It is implemented by combining the ISTA and ADMM algorithms, and works 
// for continuous, multimonial and survival data. 

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins("cpp11")]]
//[[Rcpp::interfaces(r,cpp)]]

#include <RcppArmadillo.h>
#include "penalty.h"

using namespace Rcpp;
using namespace arma;

//' Generalized linear model constraint on hierarchical structure
//' by using overlapped group penalty
//'   
//' @param y response variable, in the format of matrix. When family is 
//'          ``gaussian'' or ``binomial'', \code{y} ought to
//'          be a matrix of n by 1 for the observations; when family
//'          is ``coxph'', y represents the survival objects, that is,
//'          a matrix of n by 2, containing the survival time and the censoring status.
//'          See \code{\link[survival]{Surv}}.
//' @param x a model matrix of dimensions n by p,in which the columns represents 
//'          the predictor variables.
//' @param g a numeric vector of group labels for the predictor variables.
//' @param v a numeric vector of binary values, represents whether or not the
//'          predictor variables are penalized. Note that 1 indicates
//'          penalization and 0 for not penalization.
//' @param lambda a numeric vector of three penalty parameters corresponding to L2 norm,
//'               squared L2 norm, and L1 norm, respectively.
//' @param hierarchy a factor value in levels 0, 1, 2, which represent different
//'                  hierarchical structure within groups, respectively. 
//'                  When \code{hierarchy=0}, \eqn{\lambda_2} and \eqn{\lambda_3} are 
//'                  forced to be zeroes; when \code{hierarchy=1}, \eqn{\lambda_2} is 
//'                  forced to be zero; when \code{hierarchy=2}, there is no constraint 
//'                  on \eqn{\lambda}'s. See \code{\link{smog}}.
//' @param family a description of the distribution family for the response
//'               variable variable. For continuous response variable,
//'               family is ``gaussian''; for multinomial or binary response
//'               variable, family is ``binomial''; for survival response
//'               variable, family is ``coxph'', respectively.
//' @param subset an optional vector specifying a subset of observations to be
//'               used in the model fitting. Default is \code{NULL}.
//' @param rho the penalty parameter used in the alternating direction method
//'            of multipliers algorithm (ADMM). Default is 1e-3.
//' @param scale whether or not scale the design matrix. Default is \code{false}.
//' @param eabs  the absolute tolerance used in the ADMM algorithm. Default is 1e-3.
//' @param erel  the reletive tolerance used in the ADMM algorithm. Default is 1e-3.
//' @param LL    initial value for the Lipschitz continuous constant for
//'              approximation to the objective function in the Majorization-
//'              Minimization (MM) (or iterative shrinkage-thresholding algorithm
//'              (ISTA)). Default is 1.
//' @param eta gradient stepsize for the backtrack line search for the Lipschitz
//'            continuous constant. Default is 1.25.
//' @param maxitr the maximum iterations for convergence in the ADMM algorithm.
//'               Default is 500.
//' @keywords internal
//' 
//[[Rcpp::export]]
Rcpp::List glog(const arma::mat & y, 
                const arma::mat & x, 
                const arma::uvec & g, 
                const arma::uvec & v,
                const arma::vec & lambda,
                const int & hierarchy,
                const std::string & family = "gaussian",
                const double & rho = 1e-3,
                const bool & scale = false,
                const double & eabs = 1e-3,
                const double & erel = 1e-3,
                const double & LL = 1,
                const double & eta = 1.25,
                const int & maxitr = 500){
  
  if(family == "gaussian"){
    // centralize y and x
    const arma::vec Y0 = y.col(0);
    const arma::vec Y = scale ? Y0 - arma::mean(Y0) : Y0;
    const arma::mat X = scale ? x.each_row() - arma::mean(x) : x; 
    
    const int n = x.n_rows;
    const int p = x.n_cols;
    
    // cache the inverse matrix 
    const arma::mat I_n = arma::eye<arma::mat>(n,n);
    const arma::mat I_p = arma::eye<arma::mat>(p,p);
    const arma::mat invm = 1.0/rho*(I_p-X.t()*((rho*I_n+X*X.t()).i())*X);
    
    //initialize primal, dual and error variables
    arma::vec new_beta = arma::zeros(p);
    arma::vec old_Z = arma::zeros(p);  // augmented variable for beta
    arma::vec new_Z = arma::zeros(p); 
    arma::vec U = arma::zeros(p);      // dual variable
    
    std::vector<double> epri;   // primal error
    std::vector<double> edual;  // dual error 
    double epri_ctr = 0.0;      // primal error control
    double edual_ctr = 0.0;     // dual error control
    
    const arma::uvec idx0 = arma::find(v == 0);
    arma::uvec ug = arma::unique(g%v);
    arma::uvec upg = ug(arma::find(ug>0));
    
    std::vector<double> llike;
    arma::uvec d; 
    d << 1;
    int itr = 0;
    do{
      double lpenalty = 0.0;
      // ADMM: update the primal variable -- beta
      new_beta = invm*(X.t()*Y+rho*(new_Z-U));
      
      // ADMM: update the dual variable -- Z
      new_Z.elem(idx0) = new_beta.elem(idx0) + U.elem(idx0);
      
      arma::uvec idx1;
      for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
        idx1 = arma::find(g == (*it));
        new_Z.elem(idx1) = prox(new_beta.elem(idx1) + U.elem(idx1),lambda/rho, hierarchy, d);
        lpenalty += penalty(new_Z.elem(idx1), lambda, hierarchy, d);
      }
      
      // ADMM: update the dual variable -- U
      U=U+(new_beta - new_Z);
      
      // ADMM: Update primal and dual errors
      epri.push_back(arma::norm(new_beta - new_Z,2)/std::sqrt(p));
      edual.push_back(arma::norm(new_Z - old_Z,2)/std::sqrt(p));
      
      epri_ctr = eabs + erel/std::sqrt(p)*(arma::norm(new_beta,2) > arma::norm(new_Z,2) ? 
                                             arma::norm(new_beta,2) : arma::norm(new_Z,2));
      edual_ctr = std::sqrt(n/p)*eabs/rho + erel/std::sqrt(p)*(arma::norm(U,2));
      
      old_Z = new_Z;
      llike.push_back(0.0 - std::pow(arma::norm(Y - X*new_Z,2),2)/2 - lpenalty);
      itr++;
    } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
    
    arma::uvec subindex = arma::find(arma::abs(new_Z) > 0.0);
    arma::vec beta = new_Z.elem(subindex);
    int nsubidx = subindex.n_elem;
    
    arma::rowvec xm = arma::mean(x);
    double beta0 = subindex.is_empty() ? arma::mean(y.col(0)) : 
      arma::mean(y.col(0)) - arma::dot(xm.elem(subindex),beta); 
    
    arma::uvec Beta(nsubidx+1);
    arma::vec Coefficients(nsubidx+1);
    
    if(nsubidx == 0){
      Beta(0) = 0;
      Coefficients(0) = beta0;
    }else{
      Beta(0) = 0;
      Beta.subvec(1,nsubidx) = subindex + 1;
      
      Coefficients(0) = beta0;
      Coefficients.subvec(1,nsubidx) = beta; 
    }
    
    return Rcpp::List::create(Rcpp::Named("coefficients")=Rcpp::DataFrame::create(_["Id"] = Beta,
                                          _["Estimate"] = Coefficients),
                                          Rcpp::Named("loglike") = llike,
                                          Rcpp::Named("PrimalError") = epri,
                                          Rcpp::Named("DualError") = edual,
                                          Rcpp::Named("converge") = itr);
  }else if(family == "binomial"){
    // centralize x
    const arma::vec Y = y.col(0);
    const arma::mat X = scale ? x.each_row() - arma::mean(x) : x; 
    const arma::vec G = arma::unique(Y);  // group levels 
    
    const int K = G.n_elem;               // number of groups
    const int n = X.n_rows;               // number of observations
    const int p = X.n_cols;               // number of predictor variables
    
    // initialize primal, dual and error variables
    arma::mat new_beta = arma::zeros<arma::mat>(p,K-1);
    arma::mat old_Z = arma::zeros<arma::mat>(p,K-1);  // augmented variable for beta
    arma::mat new_Z = arma::zeros<arma::mat>(p,K-1); 
    arma::mat U = arma::zeros<arma::mat>(p,K-1);      // dual variable
    
    // calculate the gradient 
    arma::mat theta = arma::zeros<arma::mat>(n,K-1);                 // exp(x*old_beta)
    arma::mat temp_theta = arma::zeros<arma::mat>(n,K-1);            // exp(x*new_beta)
    arma::mat grad =  arma::zeros<arma::mat>(p,K-1);                 // gradient 
    arma::mat CX = arma::zeros<arma::mat>(p,K-1);
    for(int k = 0; k<K-1; ++k){
      CX.col(k) = arma::vectorise(arma::sum(X.rows(arma::find(Y==G(k)))));
    }
    
    const arma::uvec idx0 = arma::find(v == 0);  // variables for not penalization 
    arma::uvec ug = arma::unique(g%v);
    arma::uvec upg = ug(arma::find(ug>0));       // variables for penalization
    
    std::vector<double> epri;   // primal error
    std::vector<double> edual;  // dual error 
    double epri_ctr = 0.0;      // primal error control
    double edual_ctr = 0.0;     // dual error control
    
    std::vector<double> llike;
    
    arma::uvec d; 
    d << 1;
    int itr=0;
    do{
      // Majorization-Minization step for beta (use FISTA)
      // calculate the gradient
      double old_f = 0.0;
      double new_f = 0.0;
      double Q = 0.0;
      double lpenalty = 0.0;
      
      theta = arma::exp(X*new_beta);        // dictionary for loglike fun and gradient
      grad=X.t()*((1.0/(1+arma::sum(theta,1)))%theta.each_col())-CX;
      old_f = arma::accu(arma::log(1+arma::sum(theta,1)))-arma::accu(CX%new_beta);
      
      // inner loop for FASTA
      int inner_itr = 0;
      double L = LL;
      arma::mat temp_beta = arma::zeros<arma::mat>(p,K-1);
      do{
        // ADMM: update the primal variable -- beta
        temp_beta = 1.0/(L+rho)*(L*new_beta-grad+rho*(new_Z-U));
        temp_theta = arma::exp(X*temp_beta);        // dictionary for loglike fun, gradient and hessian matrix
        new_f = arma::accu(arma::log(1+arma::sum(temp_theta,1)))-arma::accu(CX%temp_beta);
        Q = old_f + arma::dot(arma::vectorise(grad),arma::vectorise(temp_beta-new_beta)) +
                  L/2*arma::norm(temp_beta-new_beta,2);
        L *= eta;
        inner_itr++;
      } while (new_f - Q > 0 && inner_itr<50);
      new_beta = temp_beta;
      
      // ADMM: update the dual variable -- Z
      arma::uvec uk;
      for(int k = 0; k<K-1; ++k){
        uk << k;
        new_Z(idx0,uk) = new_beta(idx0,uk) + U(idx0,uk);
        
        arma::uvec idx1;
        for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
          idx1 = arma::find(g == (*it));
          new_Z(idx1,uk) = prox(new_beta(idx1,uk)+U(idx1,uk),lambda/rho,hierarchy,d);
          lpenalty += penalty(new_Z(idx1,uk),lambda,hierarchy,d);
        }
      }
      
      // ADMM: update the dual variable -- U
      U=U+(new_beta - new_Z);
      
      // ADMM: Update primal and dual errors
      epri.push_back(arma::norm(arma::vectorise(new_beta - new_Z),2)/std::sqrt(p*(K-1)));
      edual.push_back(arma::norm(arma::vectorise(new_Z - old_Z),2)/std::sqrt(p*(K-1)));
      
      epri_ctr = eabs + erel/std::sqrt(p*(K-1))*(arma::norm(arma::vectorise(new_beta),2) > arma::norm(arma::vectorise(new_Z),2) ? 
                                                   arma::norm(arma::vectorise(new_beta),2) : arma::norm(arma::vectorise(new_Z),2));
      edual_ctr = std::sqrt(n/p*(K-1))*eabs/rho + erel/std::sqrt(p*(K-1))*(arma::norm(arma::vectorise(U),2));
      
      old_Z = new_Z;
      llike.push_back(0.0 - new_f - lpenalty);
      itr++;
    } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
    
    // get the selected variables and the corresponding coefficients
    arma::uvec subindex;
    arma::uvec Beta;
    arma::mat Coefficients;
    
    arma::uvec uk;
    for(int k = 0; k<K-1; ++k){
      uk << k;
      subindex = arma::join_cols(subindex,arma::find(arma::abs(new_Z.col(k)) > 0.0));
    }
    Beta = arma::unique(subindex);
    Coefficients = new_Z.rows(Beta);
    
    return Rcpp::List::create(Rcpp::Named("coefficients") = Rcpp::DataFrame::create(_["Id"] = Beta+1,
                                          _["Estimate"] = Coefficients),
                                          Rcpp::Named("loglike") = llike,
                                          Rcpp::Named("PrimalError") = epri,
                                          Rcpp::Named("DualError") = edual,
                                          Rcpp::Named("converge") = itr);
  }else if(family == "coxph"){
    // centralize x
    const arma::mat X = scale ? x.each_row() - arma::mean(x) : x; 
    
    const arma::vec time = y.col(0);
    const arma::vec censor = y.col(1);
    arma::uvec idx = arma::find(censor>0);
    arma::vec Y = arma::unique(time(idx));
    
    const int n = X.n_rows;
    const int p = X.n_cols;
    
    //initialize primal, dual and error variables
    arma::vec new_beta = arma::zeros(p);
    arma::vec old_Z = arma::zeros(p);  // augmented variable for beta
    arma::vec new_Z = arma::zeros(p); 
    arma::vec U = arma::zeros(p);      // dual variable
    
    std::vector<double> epri;   // primal error
    std::vector<double> edual;  // dual error 
    double epri_ctr = 0.0;      // primal error control
    double edual_ctr = 0.0;     // dual error control
    
    const arma::uvec idx0 = arma::find(v == 0);  // variables for not penalization 
    arma::uvec ug = arma::unique(g%v);
    arma::uvec upg = ug(arma::find(ug>0));       // variables for penalization
    
    arma::vec theta = arma::zeros(n);                 // exp(x*old_beta)
    arma::vec temp_theta = arma::zeros(n);                // exp(x*new_beta)
    arma::mat grad_dic = arma::zeros<arma::mat>(p,n);    // gradient dictionary for each observation
    arma::vec grad =  arma::zeros(p);
    
    double res;
    std::vector<double> llike;
    double L = LL;
    
    arma::uvec d; 
    d << 1;
    int itr=0;
    do{
      // Majorization-Minization step for beta (use FISTA)
      // calculate gradient and hessian matrix
      arma::uvec temp0;
      arma::uvec temp1;
      double old_f = 0.0;
      double lpenalty = 0.0;
      
      // outer loop for noncensoring observations
      theta = arma::exp(X*new_beta);        // dictionary for loglike fun, gradient and hessian matrix
      grad_dic = (X.each_col()%theta).t();
      for(arma::vec::iterator it = Y.begin(); it!=Y.end(); ++it){
        temp0 = arma::find(time == (*it));  // tied survival time with noncensoring status
        temp1 = arma::find(time > (*it));  // risk observations at time *it (include censored observations)
        
        // inner loop for tied observation
        int temp0_n = temp0.n_elem;
        double cnst = 0.0;
        for(int l = 0; l<temp0_n; ++l){
          cnst = arma::sum(theta.elem(temp0))*(1-l/temp0_n)+arma::sum(theta.elem(temp1));
          old_f += std::log(cnst);
          grad += 1.0/cnst*(arma::sum(grad_dic.cols(temp0),1)*(1-l/temp0_n) + 
            arma::sum(grad_dic.cols(temp1),1));
        }
        old_f -= arma::dot(arma::sum(X.rows(temp0)),new_beta);
        grad -= arma::vectorise(arma::sum(X.rows(temp0)));
      }
      
      // inner loop for FASTA
      int inner_itr = 0;
      arma::vec temp_beta = arma::zeros(p);
      double new_f = 0.0;
      do{
        // ADMM: update the primal variable -- beta
        temp_beta = 1.0/(L+rho)*(L*new_beta-grad+rho*(new_Z-U));
        
        arma::uvec ttemp0;
        arma::uvec ttemp1;
        double Q = 0.0;
        new_f = 0.0;
        
        // calculate the majorization value
        temp_theta = arma::exp(X*temp_beta);        // dictionary for loglike fun, gradient and hessian matrix
        for(arma::vec::iterator it = Y.begin(); it!=Y.end(); ++it){
          ttemp0 = arma::find(time == (*it));  // tied survival time with noncensoring status
          ttemp1 = arma::find(time > (*it));  // risk observations at time *it (include censored observations)
          
          // inner loop for tied observation
          int ttemp0_n=ttemp0.n_elem;
          for(int l = 0; l<ttemp0_n; ++l){
            new_f += std::log(arma::sum(temp_theta.elem(ttemp0))*(1-l/ttemp0_n)+
              arma::sum(temp_theta.elem(ttemp1)));
          }
          new_f -= arma::dot(arma::sum(X.rows(ttemp0)),new_beta);
        }
        Q = old_f + arma::dot(grad,temp_beta-new_beta)+L/2*arma::norm(temp_beta-new_beta,2);
        
        res = new_f - Q;
        L *= eta;
        inner_itr++;
      } while (res>0 && inner_itr<50);
      new_beta = temp_beta;
      
      // ADMM: update the dual variable -- Z
      new_Z.elem(idx0) = new_beta.elem(idx0) + U.elem(idx0);
      
      arma::uvec idx1;
      for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
        idx1 = arma::find(g == (*it));
        new_Z.elem(idx1) = prox(new_beta.elem(idx1)+U.elem(idx1),lambda/rho,hierarchy,d);
        lpenalty += penalty(new_Z.elem(idx1),lambda,hierarchy,d);
      }
      
      // ADMM: update the dual variable -- U
      U=U+(new_beta - new_Z);
      
      // ADMM: Update primal and dual errors
      epri.push_back(arma::norm(new_beta - new_Z,2)/std::sqrt(p));
      edual.push_back(arma::norm(new_Z - old_Z,2)/std::sqrt(p));
      
      epri_ctr = eabs + erel/std::sqrt(p)*(arma::norm(new_beta,2)>arma::norm(new_Z,2) ? 
                                             arma::norm(new_beta,2) : arma::norm(new_Z,2));
      edual_ctr = std::sqrt(n/p)*eabs/rho + erel/std::sqrt(p)*(arma::norm(U,2));
      
      old_Z = new_Z;
      llike.push_back(0.0 - new_f - lpenalty);
      itr++;
    } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
    
    // get the selected variables and the corresponding coefficients
    arma::uvec subindex = arma::find(arma::abs(new_Z) > 0.0);
    arma::uvec Beta = subindex;
    arma::vec Coefficients = new_Z.elem(subindex);
    
    return Rcpp::List::create(Rcpp::Named("coefficients")=Rcpp::DataFrame::create(_["Id"] = Beta+1,
                                          _["Estimate"] = Coefficients),
                                          Rcpp::Named("loglike") = llike,
                                          Rcpp::Named("PrimalError") = epri,
                                          Rcpp::Named("DualError") = edual,
                                          Rcpp::Named("converge") = itr);
  }else {
    Rcpp::stop("type not matched! type must be either gaussian, binomial or coxph!");
  }
  
}

