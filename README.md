# R package: smog
Structural Modeling by using Overlapped Group Penalty

## Introduction
This R package fits a linear non-penalized phynotype (demographic) variables and penalized groups of prognostic effect and predictive effect, by satisfying such hierarchy structures that if a predictive effect exists, its prognostic effect must also exist. This package can deal with continuous, binomial or multinomial, and survival response variables, underlying the assumption of Gaussian, binomial (multinomial), and cox proportional hazard models, respectively. It is implemented by combining the iterative shrinkage-thresholding algorithm (ISTA) and the alternating direction method of multipliers algorithms (ADMM). The main method is built in C++, and the complementary methods are written in R. 

## Install smog in R
* Download the zip file `smog_1.0.tar.gz`.
* In R console, run `install.packages("smog_1.0.tar.gz",repos=NULL,type="source")`. 






