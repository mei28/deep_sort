import multiprocessing
from fastprogress import progress_bar

processes = multiprocessing.cpu_count()


def multiprocess_imap(func, args, processes: int = processes, verbose=False):
    with multiprocessing.Pool(processes=processes) as pool:
        imap = pool.imap(func, args)
        if verbose:
            progress = list(progress_bar(imap, total=len(args)))
