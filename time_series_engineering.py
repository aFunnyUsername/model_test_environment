#following the realisation that I'm working with a time series problem, it's time to start bringing data in 
#with that in mind
import pandas as pd

fp = 'tmy3MSPairport.csv'
tmy = pd.read_csv(fp, header=1)
time_list = []
for i, time in enumerate(tmy['Time (HH:MM)']):
	if time == '24:00:00':
		time = '0:00'
		time_list.append(time)
	else:
		time_list.append(time)
time_list.insert(0, time_list.pop(len(time_list) - 1))
tmy['Time (HH:MM)'] = time_list

datetime_list = []

for i, date in enumerate(tmy['Date (MM/DD/YYYY)']):
	datetime = date + ' ' + time_list[i]
	datetime_list.append(datetime)

tmy['DateTime'] = datetime_list

tmy['DateTime'] = pd.to_datetime(tmy['DateTime'])

tmy_ts_df = pd.DataFrame()
tmy_ts_df['Month'] = [tmy.loc[i, 'DateTime'].month for i in range(len(tmy['DateTime']))]
tmy_ts_df['Day'] = [tmy.loc[i, 'DateTime'].day for i in range(len(tmy['DateTime']))]
tmy_ts_df['Hour'] = [tmy.loc[i, 'DateTime'].hour for i in range(len(tmy['DateTime']))]


"""
tmy = pd.read_csv(fp, header=1, parse_dates=[['Date (MM/DD/YYYY)', 'Time (HH:MM)']])
print(tmy['Date (MM/DD/YYYY)_Time (HH:MM)'])
for i in tmy['Date (MM/DD/YYYY)_Time (HH:MM)']:
	if tmy.loc[i, 'Date (MM/DD/YYYY_Time (HH:MM)'] == '24:00:00':
		tmy.loc[i, 'Date (MM/DD/YYYY)_Time (HH:MM)'] = '0:00'
tmy['Date (MM/DD/YYYY)_Time (HH:MM)'] = pd.to_datetime(tmy['Date (MM/DD/YYYY)_Time (HH:MM)'])
print(tmy['Date (MM/DD/YYYY)_Time (HH:MM)'].month)"""


























