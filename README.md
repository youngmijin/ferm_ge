**A Fair Empirical Risk Minimization with Generalized Entropy**
---------------------------------------------------------------

This repository contains an official implementation of the paper:
https://arxiv.org/abs/2202.11966

---------------------------------------------------------------

#### __REQUIREMENTS__

* Python 3.10+
* g++ supporting C++11


#### __USAGE__

1. Install dependencies

        python -m pip install .

2. Run experiments

        python main.py --study_type convergence --dataset adult --metrics ge_bar --metrics err_bar --lambda_max 20.0 --nu 0.01 --alpha 0.0 --alpha 1.0 --alpha 2.0 --gamma 0.04 --c 8.0 --a 5.0

        python main.py --study_type varying_gamma --dataset adult --metrics ge --metrics err --lambda_max 20.0 --nu 0.01 --alpha 0.0 --alpha 1.0 --alpha 2.0 --gamma "np.linspace(0.02, 0.07, 20)" --c 8.0 --a 5.0

3. Then the outputs will be saved in `./output/` directory.

Type `main.py -h` for more options.


#### __NOTE__

* This code is not optimized for space efficiency. It may require a lot of memory.
* Running time varies depending on the parameters. It may take a few seconds to an hour.
* Feel free to leave an issue if you have any questions.
