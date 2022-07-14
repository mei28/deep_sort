import multiprocessing

processes = multiprocessing.cpu_count()


def multiprocess_imap(func, args, processes: int = processes):
    with multiprocessing.Pool(processes=processes) as pool:
        for _ in pool.imap(func, args):
            pass
