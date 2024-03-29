## Julia Projects


### Module Boolean
Provides ways to compare two boolean functions.
It uses an internal representation of boolean functions as bit-vectors and 
uses macros to provide a more intuitive external representation.

### Module Cards
Provides code to play a version of poker.

### Module Cluster
Implements a version of **kmeans** clustering.
The potential novelty being:
- A larger variety of metrics.
- A novel way to do automatic cluster number detection.
- For some metrics, weights can be used to help offset **kmeans** attraction 
  to spherical clusters.

### Module DNNS
A module to do Deep Neural Net Learning.

Additional features:
- Adds a structure that can be used to represent and compute *piece-wise linear functions*.
- The module internally extends the base functions of arithmatic, 
log and exponential functions, as well as all the trigonometric functions,
to create a *Calculus of Automatic Differentiation*.
- Automatic Differentiation is extended to piece-wise linear functions.


### Module Finance
A module containing utilities for some rudimentary signal analysis.
Some of the functions deal with the "Exponential Moving Average" and its
*unbiased* statistics. Formulas were taken from the paper:
[exponential\_moving\_average.pdf](https://github.com/scottrsm/math/tree/main/pdf/exponential_moving_average.pdf).

### Module Sudoku
Solves Sudoku puzzles.

### Module Wordle
Solves the NYT Wordle puzzle.
To be successful, one has to find the hidden word by no more than six guesses.
The solver has been designed to work well on words that are
used more frequently without sacrificing overall performance.

Below we examine 
- Overall Performance -- We consider all trials, even if a "solve" took more than six guesses.
- When Successful -- We only consider the "solves" where the number of 
guesses were less than or equal to six.

The stats for the solver are (based on 3585 five letter words):
- Overall: The mean number of guesses to solve: 4.31.
- Overall: The mean number of guesses (weighted by word usage) to solve: 2.81.
- When Successful: The mean number of guesses to solve: 4.15.
- When Successful: The mean number of guesses (weighted by word frequency) to solve: 2.81.
- Percent unsuccessful: 4.63%.
- Percent unsuccessful (weighted by word usage): 0.09%.

### Module Wquantiles
A few versions of weighted quantiles with examples and comparisons 
using data parallelization. See WquantileTest.ipynb for details.

### Jupyter Notebooks

- BoolTest.ipynb      -- Tests the Boolean module.
- CardTest.ipynb      -- Tests the Cards module.
  - Comparisons of various poker hands is produced.
  - A poker game is simulated with two players.
- DNNS.ipynb          -- Fits an AutoEncoder to some simple data.
- Cluster_Test.ipynb  -- Does fits with different data sets and metrics for the Cluster module.
- WordleTest.ipynb    -- Tests the Wordle puzzle solver strategies.
- WquantileTest.ipynb -- Tests the  Wquantiles module.
                         In order to examine parallelism you must ensure
                         that your notebooks are running Julia with multiple threads.


**DEV NOTES:**

You need to configure git using the following command:

`git config --local core.hooksPath .githooks/`

With this configuration, any remote push will invoke the git pre-push hook
which will point to the BASH shell script, .githooks/pre-push, in this repository.
The script will run all tests over all modules ensuring that all tests 
pass before one can push to the remote repository.

**Documentation:**
Given that your local repo path is `<REPO>`,
you may create and view documentation for a given module, `<Module>`, 
by doing the following:
- cd `<REPO>/pkgs/<Module>/docs`
- `julia make.jl`
- Point your browser to `file://<REPO>/pkgs/<Module>/docs/build/index.html` .


