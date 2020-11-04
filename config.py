class Config:

    _desired_size_STARE = 1008
    _h_STARE = 700
    _w_STARE = 605

    _desired_size_DRIVE = 592
    _h_DRIVE = 565
    _w_DRIVE = 584

    _desired_size_CHASE = 1008
    _h_CHASE = 999
    _w_CHASE = 960

    _desired_size_DROPS = 1008
    _h_DROPS = 480
    _w_DROPS = 640

    datasets = dict()
    datasets['DRIVE'] = (_h_DRIVE, _w_DRIVE, _desired_size_DRIVE)
    datasets['STARE'] = (_h_STARE, _w_STARE, _desired_size_STARE)
    datasets['CHASE'] = (_h_CHASE, _w_CHASE, _desired_size_CHASE)
    datasets['DROPS'] = (_h_DROPS, _w_DROPS, _desired_size_DROPS)

    class Network:
        learning_rate = 1e-3
        start_neurons = 16
        keep_prob = 1
        block_size = 1
        epochs = 150
        batch_size = 2