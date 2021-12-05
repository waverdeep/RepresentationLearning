from multiprocessing import Process


def setup_multiproceesing(target, data_list):
    processes = []
    for index, data in enumerate(data_list):
        process = Process(target=target, args=(data_list,))
        processes.append(process)
    return processes


def start_multiprocessing(processes):
    for process in processes:
        process.join()