# UnbiasedML
Investigating how to train a classifier such that no peaking structure is induced in the spectrum of control variables



## Previous work

Bonsai BDT (https://arxiv.org/abs/1210.6861)

## Motivation

In particle collider physics we often want to test a composite hypothesis: Does a hypothetical particle exist and, if so, with which mass? 
This is often determined by fitting a monatonic background and a narrow Gaussian peak  (e.g. https://arxiv.org/abs/1705.03578).
Often the strength of the statistical test can be improved by selecting a specific area of phase space of auxiliary variables (e.g. decay kinematics)
The power of the test is estimated by S/sqrt(S+B), where S, B are the number of Signal, Background events, respectively. We try to optimise this metric
One way of selecting a useful area of phasespace is with machine learning algorithms. We can train on simulated data. However, for practical reasons,
we cannot simulate the signal data points at all infinite possible values of the mass. 

Imagine if we trained with a signal sample of the hypothetical particle with a mass of 300 MeV. If we gave the machine learning algorithm the mass 
as one of its variables, it would classify every data point with a mass not close to 300 MeV as background. This would be fine if a hypothetical 
particle could only exist at that mass. However, that is not the case. Now even running the machine learning algorithm on only the background
will result in a peak and the wrong acceptance of the alternate hypothesis (false "discovery"). 
This is called "sculpting the background" and is what is critical to avoid. 
What can be done here? We need some kind of regularisation that prevents the algorithm from learning something it shouldn't.

One measure is to train on signal samples with different mass hypotheses. This improves the situation, but only allows for nonzero signal estimation
close to the masses of the signal samples. 

Another step forward is to not train on the mass itself. However, the mass is related to auxiliary variables and a good classifier can learn the 
correlations, which effectively gets us back to square one.

There is another promising procedure: baking the regularisation into the loss function. Imagine plotting the fraction of accepted events as a function of mass and call it F.
If it is a horizontal line, the classifier is unbiased. However, in our situation we don't need as drastic a requirement. Loosening it can boost 
the power of the statistical test without sculping the background. 

A linear dependence of F on the mass is acceptable. Fitting a narrow Gaussian on top of an exponentially decaying background will not be affected.
Lower moments like a quadratic dependence can also acceptable depending on the signal and background model. Higher moments can sculpt the background, 
so we want to penalise them.

## Loss function

Loss function =  (L2 loss from classification) + (sum of lower moments weighted by coefficients a ) + (large number) * (residual between function and lower moments   )^2

$L = \sum^N_{i=1} (\hat{y_i}-y_i)^2 + \sum^{10}_j a_j l_j + A \sum^M_{k=1} (F_k - \sum^{10}_j l_j m_k^i)^2  $

where $y_i$ is the class of the ith point (0 for background, 1 for signal), $\hat{y_i}$ is the predicted class, $l_j$ is the jth moment and $a_j$ the corresponding relgularisation constant. A is a large number that forces higher moments to be close to zero. $F_k$ is the fraction of events passing the ML cut in bin k, $m_k^i$ is the mass in the kth bin (could use $m_k=k$. The last term is effectively the squared error between the first 10 moments and the actual distribution.

We need an orthonormal basis and can use Hermite polynomials.  
 
## To Do


figure out how to deal with empty bins

