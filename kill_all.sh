# extract pid and kill all
ps -ef | grep python | grep xuechen | grep aowquant | awk '{print $2}' | xargs kill -9