import os

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
                ppl = float(f"{float(ppl):.2f}")
                all_results[dataset] = ppl

    print(subdir, " ".join([f"{ppl}" for ppl in all_results.values()]))