import os
import pandas as pd
from itertools import product

def parse_logs(logdir):
    """
    Parse all logs in logdir, and use most recent result
    """
    logs = os.listdir(logdir)
    results = {}

    for log in sorted(logs, reverse=False):
        # use log with latest result
        with open(os.path.join(logdir, log), 'r') as f:
            # iterate through log lines
            for line in f:
                cur_result = None
                if "[PPL]" in line:
                    cur_result = line.split("[PPL]")[1]

                if cur_result:
                    dataset, ppl = cur_result.split(":")
                    dataset = dataset.strip()
                    ppl = float(f"{float(ppl):.2f}")
                    results[dataset] = ppl

    return results

def parse_first_level(    
    first_level_dir,
    first_level_type='experiment',
):
    """
    Parse first level of results
    """
    result_list = []

    for subdir in os.listdir(first_level_dir):
        result = parse_logs(os.path.join(first_level_dir, subdir))
        result[first_level_type] = subdir
        result_list.append(result)

    return result_list

def parse_second_level(
    second_level_dir,
    second_level_type='model',
    first_level_type='experiment',
):
    """
    Parse second level of results
    """
    result_list = []

    for subdir in os.listdir(second_level_dir):
        sub_result_list = parse_first_level(os.path.join(second_level_dir, subdir), 
                                            first_level_type=first_level_type)
        for sub_result in sub_result_list:
            sub_result[second_level_type] = subdir
            result_list.append(sub_result)

    # convert result list to dataframe
    df = pd.DataFrame(result_list)

    return df

def lookup_results(
    output_path,
    save_path=None,
    second_level_type='model',
    first_level_type='experiment',
):
    df = parse_second_level(output_path, 
                            second_level_type=second_level_type, 
                            first_level_type=first_level_type)
    df = df.set_index(['experiment', 'model'])
    df = df.sort_index()
    if save_path:
        df.to_csv(save_path)