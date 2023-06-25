import pandas as pd

def print_statistic(df):
    # 'x1': float(det[0]),
    # 'y1': float(det[1]),
    # 'x2': float(det[2]),
    # 'y2': float(det[3]),
    # 'conf': float(det[4]),
    # 'cls': int(det[5]),
    # 'is_moving': int(det[6]),
    # 'uuid': str(det[7]),
    unique_uuids = df['uuid'].unique()

    for uuid in unique_uuids:
        df_tmp = df[df['uuid']==uuid]
        if len(df_tmp) > 2:
            print(f'{uuid}: {df_tmp["cls"]}')
            print(df_tmp)
            # for ind, row in df_tmp.ite
            # print(df_tmp)

        print('='*50)

# df = pd.read_csv('../df_tmp.csv')
# print_statistic(df)