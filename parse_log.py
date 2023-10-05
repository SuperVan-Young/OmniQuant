import os
import pandas as pd

df = pd.DataFrame(columns=['test-name', 'wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new'])

# listdir log
for subdir in os.listdir('./log'):
    logs = os.listdir('./log/' + subdir)
    recent_log = logs[-1]
    all_results = {}
    
    with open(os.path.join('./log', subdir, recent_log), 'r') as f:
        # iterate through log lines

        for line in f:
            if "(main.py 147): INFO" in line:
                # use regex to extract str after INFO
                result = line.split("INFO")[1]
                dataset, ppl = result.split(":")
                dataset = dataset.strip()
                ppl = float(f"{float(ppl):.2f}")
                all_results[dataset] = ppl

    # print(subdir, " ".join([f"{ppl}" for ppl in all_results.values()]))
    
    # save to df
    all_results['test-name'] = subdir
    print(all_results)
    df.loc[len(df)] = all_results

with open('log.csv', 'w') as f:
    df.to_csv(f, index=False)