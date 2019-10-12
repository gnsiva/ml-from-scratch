# Machine Learning Algorithms From Scratch

Here are basic implementations of several machine learning and machine learning related algorithms.
The aim of making these was to try to better understand how they work internally, and they are not intended to be highly optimised solutions.

## Algorithms and status

- `decisiontrees`
    - Decision trees
        - Working, but would like to re-write with recursion.
    - Random forest
        - Working well
    - Gradient boosted trees
        - GradientBoostingRegressor working nicely
        - GradientBoostingMAERegressor not performing properly
            - Unittest vs sklearn version currently commented out
- `featureimportances`
    - Partial dependence plots (pdplots)
    - Permutation importance
- `naivebayes`
    - Bernoulli naive Bayes
        - Working well, same results as sklearn version
    - Multinomial naive Bayes
        - Working well, same results as sklearn version
    - Discrete naive Bayes
        - Working
    - Gaussian naive Bayes
        - Working, same prediction as sklearn version
 