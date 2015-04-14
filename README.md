I don't (as of the start of this project) know how to use scikit-learn or pandas
well at all. Let's fix that.

Here's the setup: I have a dataset from a 1994 Census survey of individuals, with
demographic and economic traits, and a binary indicator for whether the individual made
more than $50,000 a year. I've taken this dataset, explored and cleaned it, and
set up a model bake-off to see how well I can predict the high-income indicator.
I'll conduct the same analysis a) in R, b) with pandas / scikit-learn in Python.

Let's use these measures of performance:
  o) inspecting a confusion matrix
  o) the ROC curve, plotted
  o) the area under the ROC curve

This is much more about me picking up the scikit-learn API than it is about good
predictive performance :-)

TODO:
  o) this should be actual markdown
  o) the R should use caret
