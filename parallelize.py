import ray
try:
    from satpy_tools.constants import MEM_PER_WORKER, CPUS
except ImportError:
    from constants import MEM_PER_WORKER, CPUS
import numpy as np
from time import sleep


def smart_parallelize(func, n_vars2parallelize=1): 
    '''FIRST ARG MUST BE ARG TO BE PARALLELIZED'''

    def wrap(**kwargs): 
        @ray.remote(memory=MEM_PER_WORKER)
        def get_results(**kwargs):
            try:
                args_names = ray.get(args_names_put)
                args_vals  = ray.get(args_vals_put)
                kwargs2 = dict(zip(args_names, args_vals))
                kwargs.update(kwargs2)
            except NameError:
                pass

            return func(**kwargs)
        
        arg2parallelize = kwargs[list(kwargs.keys())[0]]
        # split into CPUS chunks
        arg2parallelize = np.array_split(arg2parallelize, CPUS)

        arg_names = list(kwargs.keys())[1:]
        arg_vals  = list(kwargs.values())[1:]

        if len(arg_names) != 0:
            args_names_put = ray.put(arg_names)
            args_vals_put  = ray.put(arg_vals)

        workers = []
        for i in range(len(arg2parallelize)):
            par_arg    = {list(kwargs.keys())[0]:arg2parallelize[i]}
            workers.append(get_results.remote(**par_arg))
        result = ray.get(workers)
        result = np.concatenate(result)

        return result 
    return wrap 


if __name__ == "__main__":
    import timeit

    @smart_parallelize
    def func(x, y):
        sleep(1)
        return x + y
    
    x = np.arange(10)
    y = 64
    start  = timeit.default_timer()
    answer = func(x=x, y=y)
    stop  = timeit.default_timer()
    print(stop - start)

    print(answer)

    start  = timeit.default_timer()
    answer = []
    for i, val in enumerate(x):
        sleep(1)
        answer.append(val+y)
    stop = timeit.default_timer()
    print(stop - start)

    print(answer)
