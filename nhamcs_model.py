import pandas as pd
from sas7bdat import SAS7BDAT

# loop over range 2015 to 2020
for year in range(2015, 2021):
  with SAS7BDAT(f'ed{year}_sas.sas7bdat') as file:
    df = file.to_data_frame()

    # ADMIT descriptions:
    # 9 = Blank
    # -8 = Data not available (Unknown)
    # -7 = Not applicable (not admitted to hospital)
    # 1 = Critical care unit
    # 2 = Stepdown unit
    # 3 = Operating room
    # 4 = Mental health or detox unit
    # 5 = Cardiac catheterization lab
    # 6 = Other bed/uni

    # drow rows where ADMIT does not equal -7 or 1

    df = df[(df['ADMIT'] == -7) | (df['ADMIT'] == 1)]

    # replace -7 with 0

    df['ADMIT'] = df['ADMIT'].replace(-7, 0)

    # Features
    X = df[['AGE', 'SEX', 'ARREMS', 'TEMPF', 'PULSE', 'BPSYS',
              'BPDIAS', 'RESPR', 'POPCT',
                'RFV1', 'NOCHRON', 'TOTCHRON']]

    # Labels
    y = df[['ADMIT']]

    # save X to csv
    X.to_csv(f'features_{year}.csv', index=False)

    # save y to csv
    y.to_csv(f'labels_{year}.csv', index=False)