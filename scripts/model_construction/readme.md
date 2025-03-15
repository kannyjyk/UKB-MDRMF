# Model constuction

### Demo data training
1. Open `randomdatademo.ipynb`

2. Execute each block in its order.

This demo will walk you through a simlified process of our code.


### Real data training
1. Check GPU info and locations in `config.py` first.

2. Run `cacheX.ipynb` to generate blockwise data.

3. Review `trainwrapper.py` for the tasks you wanted to implement.

Note: We have prepared scripts for the tasks in our paper, comment and change the parameter list to remove task you don't want.

Integer for tasks, starts from 0:
dummy, standard, survival, priority, survival priority, sub category, survival sub category, importance, special

4. Execute `trainwrapper.py 1` and `run.py` using your Python interpreter.








