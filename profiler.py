import cProfile
import pstats
import os

def profiler(function, *args, **kwargs):
    with cProfile.Profile() as pr:
        function(*args, **kwargs)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats('profile.prof')
    os.system('snakeviz ./profile.prof')

    os.remove('profile.prof')

    return stats

