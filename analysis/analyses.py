def find_records(date=None, run=None):
    if date is None:
        date = str(datetime.date.today())
    date_path = root_path + date + '/'
    
    if run is None:
        run = get_last_file_in_dir(date_path)
    run_path = date_path + str(run) + '/'
    
    part_params_filename = run_path + 'part_params.json'
    with open(part_params_filename, mode='r') as part_params_file:
        part_params = json.load(part_params_file)

    wall_params_filename = run_path + 'wall_params.json'
    with open(wall_params_filename, mode='r') as wall_params_file:
        wall_params = json.load(wall_params_file)

    data_filename = run_path + 'data.hdf5'
    return part_params, wall_params, data_filename, run_path