import os
import pandas as pd

os.makedirs("./results", exist_ok=True)

def collect_results() -> pd.DataFrame:

    df = pd.DataFrame(columns=['experiment', 'model', 'wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new'])

    output_dir = './output'

    # listdir log
    for experiment in os.listdir(output_dir):
        models = os.listdir(os.path.join(output_dir, experiment))
        models = sorted(models)

        for model in models:
            logs = os.listdir(os.path.join(output_dir, experiment, model))

            for log in sorted(logs):
                # use log with latest result
            
                with open(os.path.join(output_dir, experiment, model, log), 'r') as f:
                    # iterate through log lines

                    result = {
                        'experiment': experiment, 
                        'model': model,
                    }

                    for line in f:
                        cur_result = None
                        if "(main.py 148): INFO" in line:
                            # use regex to extract str after INFO
                            cur_result = line.split("INFO")[1]
                            
                        elif "[PPL]" in line:
                            cur_result = line.split("[PPL]")[1]

                        if cur_result:
                            dataset, ppl = cur_result.split(":")
                            dataset = dataset.strip()
                            ppl = float(f"{float(ppl):.2f}")
                            result[dataset] = ppl


                # print(subdir, " ".join([f"{ppl}" for ppl in all_results.values()]))
                
                # find previous result in df
                # if found, update row
                # else, create new row
                mask = (df['experiment'] == experiment) & (df['model'] == model)
                if len(df.loc[mask]) > 0:
                    # update non-empty items
                    for key,value in result.items():
                        if key in df.columns and value is not None:
                            df.loc[mask, key] = value
                else:
                    df.loc[len(df)] = result

    return df

def lookup_results(
    df, 
    experiment_list, 
    model_list,
    save_path=None,
):
    """
    Lookup results from df.
    """
    # set index to experiment and model
    df = df.set_index(['experiment', 'model'])

    # filter by experiment and model
    mask = (df.index.get_level_values('experiment').isin(experiment_list)) & (df.index.get_level_values('model').isin(model_list))
    filtered_df = df.loc[mask]

    sorted_df = filtered_df.sort_values(['model', 'experiment'])

    if save_path:
        with open(save_path, 'w') as f:
            sorted_df.to_csv(f, index=True)

    return sorted_df

MODEL_LIST = [
    'opt-6.7b', 
    'opt-13b', 
    'opt-30b', 
    'opt-66b', 
    'llama-7b-meta', 
    'llama-13b-meta',
    'llama-30b-meta',
    'llama-65b-meta',
]

LOOKUPS = [
    {
        'experiment_list': ['W16A16', 'qkvproj_W16A4', 'qkvproj_W16A8', 'qkvproj_W16A4_ol0.01'],
        'model_list': MODEL_LIST,
        'save_path': 'results/qkvproj.csv'
    },
    {
        'experiment_list': ['W16A16', 'oproj_W16A4', 'oproj_W16A8', 'oproj_W16A4_g128', 'oproj_W16A4_ol1', 'oproj_W16A4_g128_ol1'],
        'model_list': MODEL_LIST,
        'save_path': 'results/oproj.csv'
    },
    {
        'experiment_list': ['W16A16', 'fc1_W16A4', 'fc1_W16A8', 'fc1_W16A4_ol0.01'],
        'model_list': MODEL_LIST,
        'save_path': 'results/fc1.csv'
    },
    {
        'experiment_list': ['W16A16', 'fc2_W16A4', 'fc2_W16A8', 'fc2_W16A4_g128', 'fc2_W16A4_ol1', 'fc2_W16A4_g128_ol1'],
        'model_list': MODEL_LIST,
        'save_path': 'results/fc2.csv'
    },
    {
        'experiment_list': ['W16A16', 'q_W16A4', 'q_W16A8'],
        'model_list': MODEL_LIST,
        'save_path': 'results/q.csv'
    },
    {
        'experiment_list': ['W16A16', 'k_W16A4', 'k_W16A8'],
        'model_list': MODEL_LIST,
        'save_path': 'results/k.csv'
    },
    {
        'experiment_list': ['W16A16', 'v_W16A4', 'v_W16A8'],
        'model_list': MODEL_LIST,
        'save_path': 'results/v.csv'
    },
]

if __name__ == '__main__':
    df = collect_results()
    # with open('log.csv', 'w') as f:
    #     df.to_csv(f, index=False)

    for lookup in LOOKUPS:
        lookup_results(
            df, 
            experiment_list=lookup['experiment_list'], 
            model_list=lookup['model_list'],
            save_path=lookup['save_path'],
        )

