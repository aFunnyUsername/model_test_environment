#following the realisation that I'm working with a time series problem, it's time to start bringing data in 
#with that in mind
import pandas as pd
from pandas.plotting import lag_plot
from matplotlib import pyplot as plt
#NOTE here is the time scrubbing.  TMY for some reason saves there time in hours 1-24 instead of 0-23 like 
#normal people so we have to fix it.
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
#lines up the dates with times yep
for i, date in enumerate(tmy['Date (MM/DD/YYYY)']):
	datetime = date + ' ' + time_list[i]
	datetime_list.append(datetime)

tmy['DateTime'] = datetime_list
#NOTE converting the elements of the pandas series to datetime instead of string so we can parse out
#the month, time and hour later
tmy['DateTime'] = pd.to_datetime(tmy['DateTime'])

#NOTE and here we make our new dataframe with the months, days, and hours
tmy_ts_df = pd.DataFrame()
tmy_ts_df['Month'] = [tmy.loc[i, 'DateTime'].month for i in range(len(tmy['DateTime']))]
tmy_ts_df['Day'] = [tmy.loc[i, 'DateTime'].day for i in range(len(tmy['DateTime']))]
tmy_ts_df['Hour'] = [tmy.loc[i, 'DateTime'].hour for i in range(len(tmy['DateTime']))]
tmy_ts_df['Clouds'] = tmy['TotCld (tenths)'].values
tmy_ts_df['Temperature'] = tmy['Dry-bulb (C)'].values
tmy_ts_df['Humidity'] = tmy['RHum (%)'].values
tmy_ts_df['DNI'] = tmy['DNI (W/m^2)'].values
tmy_ts_df['DHI'] = tmy['DHI (W/m^2)'].values

cld_ser = tmy_ts_df['Clouds']
tem_ser = tmy_ts_df['Temperature']
hum_ser = tmy_ts_df['Humidity']
dni_ser = tmy_ts_df['DNI']
dhi_ser = tmy_ts_df['DHI']

cld_ts_plt = cld_ser.plot(linewidth=0.5)
plt.title('Clouds vs. Time')
plt.xlabel('Time (hours)')
plt.ylabel('Cloud Cover (tenths)')
plt.savefig('plots\\cldvstime_line_07012018.png')
plt.close()
tem_ts_plt = tem_ser.plot(linewidth=0.5)
plt.title('Temperature vs. Time')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (degrees C)')
plt.savefig('plots\\temvstime_line_07012018.png')
plt.close()
hum_ts_plt = hum_ser.plot(linewidth=0.5)
plt.title('Humidity vs. Time')
plt.xlabel('Time (hours)')
plt.ylabel('Humidity (%)')
plt.savefig('plots\\humvstime_line_07012018.png')
plt.close()
dni_ts_plt = dni_ser.plot(linewidth=0.5)
plt.title('DNI vs. Time')
plt.xlabel('Time (hours)')
plt.ylabel('Direct Normal Irradiance (W/m^2)')
plt.savefig('plots\\dnivstime_line_07012018.png')
plt.close()
dhi_ts_plt = dhi_ser.plot(linewidth=0.5)
plt.title('DHI vs. Time')
plt.xlabel('Time (hours)')
plt.ylabel('Diffuse Horizontal Irradiance (W/m^2)')
plt.savefig('plots\\dhivstime_line_07012018.png')
plt.close()

cld_ts_hist = cld_ser.hist(bins=10)
plt.title('Clouds')
plt.xlabel('Cloud Cover (tenths)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\cldvstime_hist_07012018.png')
plt.close()
tem_ts_hist = tem_ser.hist(bins=100)
plt.title('Temperature')
plt.xlabel('Temperature (degrees C)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\temvstime_hist_07012018.png')
plt.close()
hum_ts_hist = hum_ser.hist(bins=100)
plt.title('Humidity')
plt.xlabel('Humidity (%)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\humvstime_hist_07012018.png')
plt.close()
dni_ts_hist = dni_ser.hist(bins=100)
plt.title('DNI')
plt.xlabel('Direct Normal Irradiance (W/m^2)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\dnivstime_hist_07012018.png')
plt.close()
dhi_ts_hist = dhi_ser.hist(bins=100)
plt.title('DHI')
plt.xlabel('Diffuse Horizontal Irradiance (W/m^2)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\dhivstime_hist_07012018.png')
plt.close()

cld_ts_dens = cld_ser.plot(kind='kde')
plt.title('Clouds')
plt.xlabel('Cloud Coverage (Tenths)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\cldvstime_dens_07012018.png')
plt.close()
tem_ts_dens = tem_ser.plot(kind='kde')
plt.title('Temperature')
plt.xlabel('Temperature (C)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\temvstime_dens_07012018.png')
plt.close()
hum_ts_dens = hum_ser.plot(kind='kde')
plt.title('Humidity')
plt.xlabel('Humidity (%)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\humvstime_dens_07012018.png')
plt.close()
dni_ts_dens = dni_ser.plot(kind='kde')
plt.title('DNI')
plt.xlabel('Direct Normal Irradiance (W/m^2)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\dnivstime_dens_07012018.png')
plt.close()
dhi_ts_dens = dhi_ser.plot(kind='kde')
plt.title('DHI')
plt.xlabel('Diffuse Horizontal Irradiance (W/m^2)')
plt.ylabel('Percentage of Distribution')
plt.savefig('plots\\dhivstime_dens_07012018.png')
plt.close()

#checkin out a lag plot for both our DNI and DHI

lag_plot(dni_ser, size=1)
plt.savefig('plots\\dni_lagplot_t1_07012018.png')
plt.close()
lag_plot(dhi_ser, size=1)
plt.savefig('plots\\dhi_lagplot_t1_07012018.png')
plt.close()
#NOTE, this is really interesting, as it shows that DNI's correlation with previous observations gets better
#as DNI gets larger.  This makes sense, since a DNI of 700 or more means we're probably in the middle of the 
#day, so the next time is much more likely to have a high DNI as well.  In the morning and evening however, 
#there can be a high or low DNI, followed by zero (or led, in the morning) which makes the correlation weaker
#for those mid to low level numbers.  
#NOTE, dhi seems to go the other direction, but the correlation doesn't get as weak at the high DHI.  However, 
#the overall correlation is never as strong as high DNI.  Most likely, this is because DHI is heavily influenced
#by cloud cover, and even moving clouds, meaning that it's going to be much more variable in general.



#plt.show()



























