import math

minmax_dict =  {
    "Tg": [-148.0297376, 472.25],
    "FFV": [0.2269924, 0.77709707],
    "Tc": [0.0465, 0.524],
    "Density": [0.748691234, 1.840998909],
    "Rg": [9.7283551, 34.672905605],
    "He": [-2.9040650850281673, 9.786953736280177],
    "H2": [-3.780994743021742, 10.513253124157023],
    "O2": [-14.172185501903007, 9.836278802842678],
    "N2": [-8.740336742730447, 9.717157974344635],
    "CO2": [-13.633189001170319, 10.757902880692196],
    "CH4": [-6.3771270279199666, 10.46310334047155],
}

def scaling_y(y, task_name):
    if task_name in ["He", "H2", "O2", "N2", "CO2", "CH4"]:
        y = math.log(y)
    return y


def minmax_scale(y, min_val, max_val):
    y = (y - min_val) / (max_val - min_val)
    return y


def minmax_scale_inverse(y, min_val, max_val):
    y = y * (max_val - min_val) + min_val
    return y


def scaling_error(error, task_idx):
    task_name = list(minmax_dict.keys())[task_idx]
    min_val, max_val = minmax_dict[task_name]
    label_range = max_val - min_val
    return error / label_range
