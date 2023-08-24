## Julia Projects

### Module Cards
Provides code to play a version of poker.

### Module Boolean
Provides ways to compare two boolean functions.
It uses an internal representation of boolean functions as bit-vectors and 
uses macros to provide a more intuitive external representation.

### Module Wordle
Solves the NYT Wordle puzzle.

### Module Wquantiles
A few versions of weighted quantiles with examples and comparisons 
of using data parallelization.

### Jupyter Notebooks

- BoolTest.ipynb      -- Jupyter notebook to test the Boolean module.
- CardTest.ipynb      -- Jupyter notebook to test the Cards module.
  - Comparisons of various poker hands is produced.
  - A poker game is simulated with two players.
- WordleTest.ipynb    -- Jupyter notebook to test Wordle puzzle solver strategies.
- WquantileTest.ipynb -- Jupyter notebook to test Wquantiles module.
                         In order to examine parallelism you must ensure
                         that your notebooks are running julia with multiple threads.


