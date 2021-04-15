def check_params_exist(esitmator, params_keyword):
    all_params = esitmator.get_params().keys()
    available_params = [x for x in all_params if params_keyword in x]
    if len(available_params)==0:
        return "No matching params found!"
    else:
        return available_params

##ex: print(check_params_exist(pipe, 'ct'))