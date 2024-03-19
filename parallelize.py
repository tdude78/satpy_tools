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
        def f2(**kwargs):
            kwargs2       = kwargs.copy()
            kw1_name      = list(kwargs.keys())[0]
            kw1_vals      = kwargs[kw1_name]
            kwargs2[kw1_name] = kw1_vals[0]
            r0 = func(**kwargs2)
            if isinstance(r0, tuple):
                results = ()
                for i, r in enumerate(r0):
                    r = np.atleast_1d(r)
                    results += (np.zeros((len(kw1_vals), *r.shape)),)
                    tup = True
            else:
                results = np.zeros((len(kw1_vals), *r0.shape))
                tup = False

            for i, kw1_val in enumerate(kw1_vals):
                kwargs2[kw1_name] = kw1_val
                r_all = func(**kwargs2)
                if tup:
                    for j, r in enumerate(r_all):
                        results[j][i] = r
                else:
                    results[i] = r_all
            return results
        @ray.remote(memory=MEM_PER_WORKER)
        def get_results(**kwargs):
            try:
                args_names = ray.get(args_names_put)
                args_vals  = ray.get(args_vals_put)
                kwargs2 = dict(zip(args_names, args_vals))
                kwargs.update(kwargs2)
            except NameError:
                pass

            return f2(**kwargs)
        
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

        if isinstance(result[0], tuple):
            result = [list(r) for r in result]
            r2 = list(result[0])
            for i,r in enumerate(result):
                if i == 0:
                    continue
                for j, r2i in enumerate(r):
                    r2[j] = np.append(r2[j], r2i, axis=0)
            for i, r2i in enumerate(r2):
                r2[i] = np.concatenate(r2i)
            result = tuple(r2)
        else:
            result = np.concatenate(result)

        return result 
    return wrap 


if __name__ == "__main__":
    import timeit

    @smart_parallelize
    def func(x, y):
        sleep(1)
        return x,y
    
    x = np.ones((15,6))
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

