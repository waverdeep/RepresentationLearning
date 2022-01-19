from multiprocessing import Process


def setup_multiproceesing(target, data_list):
    processes = []
    for index, data in enumerate(data_list):
        print("make process #{}".format(index))
        process = Process(target=target, args=(data,))
        processes.append(process)
        process.start()
    return processes


def start_multiprocessing(processes):
    for process in processes:
        process.join()