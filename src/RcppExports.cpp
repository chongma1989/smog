// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/smog.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <string>
#include <set>

using namespace Rcpp;

// glog
Rcpp::List glog(const arma::mat& y, const arma::mat& x, const arma::uvec& g, const arma::uvec& v, const arma::vec& lambda, const int& hierarchy, const std::string& family, const double& rho, const bool& scale, const double& eabs, const double& erel, const double& LL, const double& eta, const int& maxitr);
static SEXP _smog_glog_try(SEXP ySEXP, SEXP xSEXP, SEXP gSEXP, SEXP vSEXP, SEXP lambdaSEXP, SEXP hierarchySEXP, SEXP familySEXP, SEXP rhoSEXP, SEXP scaleSEXP, SEXP eabsSEXP, SEXP erelSEXP, SEXP LLSEXP, SEXP etaSEXP, SEXP maxitrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type g(gSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type v(vSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const int& >::type hierarchy(hierarchySEXP);
    Rcpp::traits::input_parameter< const std::string& >::type family(familySEXP);
    Rcpp::traits::input_parameter< const double& >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< const bool& >::type scale(scaleSEXP);
    Rcpp::traits::input_parameter< const double& >::type eabs(eabsSEXP);
    Rcpp::traits::input_parameter< const double& >::type erel(erelSEXP);
    Rcpp::traits::input_parameter< const double& >::type LL(LLSEXP);
    Rcpp::traits::input_parameter< const double& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< const int& >::type maxitr(maxitrSEXP);
    rcpp_result_gen = Rcpp::wrap(glog(y, x, g, v, lambda, hierarchy, family, rho, scale, eabs, erel, LL, eta, maxitr));
    return rcpp_result_gen;
END_RCPP_RETURN_ERROR
}
RcppExport SEXP _smog_glog(SEXP ySEXP, SEXP xSEXP, SEXP gSEXP, SEXP vSEXP, SEXP lambdaSEXP, SEXP hierarchySEXP, SEXP familySEXP, SEXP rhoSEXP, SEXP scaleSEXP, SEXP eabsSEXP, SEXP erelSEXP, SEXP LLSEXP, SEXP etaSEXP, SEXP maxitrSEXP) {
    SEXP rcpp_result_gen;
    {
        Rcpp::RNGScope rcpp_rngScope_gen;
        rcpp_result_gen = PROTECT(_smog_glog_try(ySEXP, xSEXP, gSEXP, vSEXP, lambdaSEXP, hierarchySEXP, familySEXP, rhoSEXP, scaleSEXP, eabsSEXP, erelSEXP, LLSEXP, etaSEXP, maxitrSEXP));
    }
    Rboolean rcpp_isInterrupt_gen = Rf_inherits(rcpp_result_gen, "interrupted-error");
    if (rcpp_isInterrupt_gen) {
        UNPROTECT(1);
        Rf_onintr();
    }
    bool rcpp_isLongjump_gen = Rcpp::internal::isLongjumpSentinel(rcpp_result_gen);
    if (rcpp_isLongjump_gen) {
        Rcpp::internal::resumeJump(rcpp_result_gen);
    }
    Rboolean rcpp_isError_gen = Rf_inherits(rcpp_result_gen, "try-error");
    if (rcpp_isError_gen) {
        SEXP rcpp_msgSEXP_gen = Rf_asChar(rcpp_result_gen);
        UNPROTECT(1);
        Rf_error(CHAR(rcpp_msgSEXP_gen));
    }
    UNPROTECT(1);
    return rcpp_result_gen;
}
// proxL1
double proxL1(const double& x, const double& lambda);
RcppExport SEXP _smog_proxL1(SEXP xSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const double& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const double& >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(proxL1(x, lambda));
    return rcpp_result_gen;
END_RCPP
}
// proxL2
arma::vec proxL2(const arma::vec& x, const double& lambda);
RcppExport SEXP _smog_proxL2(SEXP xSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const double& >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(proxL2(x, lambda));
    return rcpp_result_gen;
END_RCPP
}
// prox
arma::vec prox(const arma::vec& x, const arma::vec& lambda, const int& hierarchy, const arma::uvec& d);
RcppExport SEXP _smog_prox(SEXP xSEXP, SEXP lambdaSEXP, SEXP hierarchySEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const int& >::type hierarchy(hierarchySEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(prox(x, lambda, hierarchy, d));
    return rcpp_result_gen;
END_RCPP
}
// penalty
double penalty(const arma::vec& x, const arma::vec& lambda, const int& hierarchy, const arma::uvec& d);
RcppExport SEXP _smog_penalty(SEXP xSEXP, SEXP lambdaSEXP, SEXP hierarchySEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const int& >::type hierarchy(hierarchySEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(penalty(x, lambda, hierarchy, d));
    return rcpp_result_gen;
END_RCPP
}

// validate (ensure exported C++ functions exist before calling them)
static int _smog_RcppExport_validate(const char* sig) { 
    static std::set<std::string> signatures;
    if (signatures.empty()) {
        signatures.insert("Rcpp::List(*glog)(const arma::mat&,const arma::mat&,const arma::uvec&,const arma::uvec&,const arma::vec&,const int&,const std::string&,const double&,const bool&,const double&,const double&,const double&,const double&,const int&)");
    }
    return signatures.find(sig) != signatures.end();
}

// registerCCallable (register entry points for exported C++ functions)
RcppExport SEXP _smog_RcppExport_registerCCallable() { 
    R_RegisterCCallable("smog", "_smog_glog", (DL_FUNC)_smog_glog_try);
    R_RegisterCCallable("smog", "_smog_RcppExport_validate", (DL_FUNC)_smog_RcppExport_validate);
    return R_NilValue;
}

static const R_CallMethodDef CallEntries[] = {
    {"_smog_glog", (DL_FUNC) &_smog_glog, 14},
    {"_smog_proxL1", (DL_FUNC) &_smog_proxL1, 2},
    {"_smog_proxL2", (DL_FUNC) &_smog_proxL2, 2},
    {"_smog_prox", (DL_FUNC) &_smog_prox, 4},
    {"_smog_penalty", (DL_FUNC) &_smog_penalty, 4},
    {"_smog_RcppExport_registerCCallable", (DL_FUNC) &_smog_RcppExport_registerCCallable, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_smog(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
