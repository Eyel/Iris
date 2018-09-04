import warnings
warnings.filterwarnings('ignore')


def attribute_to_list(path_csv,header=True):
    from re import compile, IGNORECASE, sub
    from pandas import read_csv

    # Import data
    if header:
        df = read_csv(path_csv, sep="#;:#")
        nom_column = list(df.columns.values)[0]
    else:
        df = read_csv(path_csv, sep="#;:#", names=['attributes'])
        nom_column = 'attributes'

    # Cleaning
    pattern1 = compile(r'(.*):.*', flags=IGNORECASE)
    pattern2 = compile(r'.*\. *([a-z0-9A-Z])', flags=IGNORECASE)
    df = df.assign(attributes_2=[sub(pattern1, r'\1', row) for row in df[nom_column]])
    df = df.assign(attributes_new=[sub(pattern2, r'\1', row) for row in df['attributes_2']])

    return list(df['attributes_new'])




