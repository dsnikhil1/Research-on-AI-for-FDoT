import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import os
from pandas import Timestamp
import pymysql

def aggregator(ramp_id, t, start_time, end_time):
    def preprocess_loop_detector_data(df):
        df_event_detail = df

        df_event_detail = df_event_detail.drop(axis=1, columns=[' '])
        dict1 = {}
        for column in df_event_detail.columns:
            dict1[column] = column.strip()
        df_event_detail = df_event_detail.rename(columns=dict1)

        df_event_detail['Timestamp'] = df_event_detail['Timestamp'].apply(lambda x: x.strip())
        df_event_detail['Category'] = df_event_detail['Category'].apply(lambda x: x.strip())
        df_event_detail['Type'] = df_event_detail['Type'].apply(lambda x: x.strip())
        df_event_detail['Timestamp'] = pd.to_datetime(df_event_detail['Timestamp'])
        df_event_detail = df_event_detail.rename(columns={'EventValue':'Event_value'})
        df_event_detail = df_event_detail[(df_event_detail['Timestamp']>=Timestamp(start_time))&(df_event_detail['Timestamp']<=Timestamp(end_time))].reset_index(drop=True)
        
        return df_event_detail
    
    def preprocess_microwave_detector_data(df):
        #df['measurement_start_timestamp'] = df['measurement_start'].apply(lambda x:str(x.split('T')[0]) + str(' ')+str(x.split('T')[1].split('-')[0]))
        df = df.dropna(subset=['speed', 'volume', 'occupancy'])
        df = df[(df['speed']>=0) & (df['volume']>=0) & (df['occupancy']>=0)]
        df['measurement_start_timestamp'] = pd.to_datetime(df['measurement_start_timestamp'])
        df = df.reset_index(drop=True)
        return df

    def aggregate_microwave_detector_data(df,t,lane_number):
        df_average_speed = pd.DataFrame({'Average Speed':df.groupby(pd.Grouper(key='measurement_start_timestamp',freq='{}min'.format(t))).speed.mean()}).reset_index()

        df_max_speed = pd.DataFrame({'Max Speed':df.groupby(pd.Grouper(key='measurement_start_timestamp',freq='{}min'.format(t))).speed.max()}).reset_index()


        df_average_flow = pd.DataFrame({'Average Flow':df.groupby(pd.Grouper(key='measurement_start_timestamp',freq='{}min'.format(t))).volume.sum()}).reset_index()


        df_average_occupancy = pd.DataFrame({'Average Occupancy':df.groupby(pd.Grouper(key='measurement_start_timestamp',freq='{}min'.format(t))).occupancy.mean()}).reset_index()
        df_aggregate = pd.merge(df_average_speed, df_average_flow, on='measurement_start_timestamp',how='left')
        df_aggregate = pd.merge(df_aggregate, df_average_occupancy, on='measurement_start_timestamp',how='left')
        df_aggregate = pd.merge(df_aggregate, df_max_speed, on='measurement_start_timestamp',how='left')

        df_aggregate['Average Flow Rate'] = df_aggregate['Average Flow'].apply(lambda x:(x*60)/t)
        df_aggregate['Congestion measure'] = df_aggregate['Average Speed']/df_aggregate['Max Speed']
        df_aggregate = df_aggregate[(df_aggregate['measurement_start_timestamp']>=Timestamp(start_time))&(df_aggregate['measurement_start_timestamp']<=Timestamp(end_time))].reset_index(drop=True)
        column_names = ['Timestamp']
        all_columns = list(df_aggregate.columns)
        for name in range(1,len(all_columns)):
            column_names.append(all_columns[name]+ ' Lane {}'.format(lane_number))
        df_aggregate.columns = column_names
        #print(df_aggregate)
        return df_aggregate

    def green_time_calculator(df,start,end):
        total_green_time = 0
        start_time = 0
        end_time = 0
        for i in range(len(df)):
            if df['Type'][i] == 'Ramp Meter - Begin Red':
                start_time = df['Timestamp'][i]
            else:
                if start_time != 0:
                    end_time = df['Timestamp'][i]
                    total_green_time += (start_time - end_time).total_seconds()

        #If last event is Begin Red, add green time from start till first event
        if df['Type'][len(df)-1] == 'Ramp Meter - Begin Red':
            total_green_time += (df['Timestamp'][len(df)-1] - start).total_seconds()

        #If first event is Begin Green, add green time from last event till end
        if df['Type'][0] == 'Ramp Meter - Begin Green':
            total_green_time += (end - df['Timestamp'][0]).total_seconds()

        return(total_green_time)

    def aggregate_loop_detector_data(df_event_detail, t):
        #df_gr is Aggregate for Ramp metering rate 
        df_gr = pd.DataFrame({'Aggregate ramp metering rate':df_event_detail.query('Type == "Ramp Meter - Lane 1 Active Rate"').groupby([pd.Grouper(key='Timestamp',freq='{}min'.format(t)),'Type']).Event_value.mean()}).reset_index()
        
        #df_gr2 is Mainline (Lane adjacent to ramp) flow rate
        df_gr2 = pd.DataFrame({'Aggregate Mainline num vehicles':df_event_detail.query('Type == "Detector On" & Event_value==11').groupby([pd.Grouper(key='Timestamp',freq='{}min'.format(t)),'Type']).Event_value.count()}).reset_index()
        df_gr2['Aggregate Mainline flow rate'] = df_gr2['Aggregate Mainline num vehicles'].apply(lambda x: x*(60/t))
        
        #df_gr4 is Ramp Demand flow rate
        df_gr4 = pd.DataFrame({'Aggregate Ramp Demand num vehicles':df_event_detail.query('Type == "Detector On" & Event_value==21').groupby([pd.Grouper(key='Timestamp',freq='{}min'.format(t)),'Type']).Event_value.count()}).reset_index()
        df_gr4['Aggregate Ramp Demand flow rate'] = df_gr4['Aggregate Ramp Demand num vehicles'].apply(lambda x: x*(60/t))
        
        #df_gr5 is Ramp Passage flow rate
        df_gr5 = pd.DataFrame({'Aggregate Ramp Passage num vehicles':df_event_detail.query('Type == "Detector On" & Event_value==23').groupby([pd.Grouper(key='Timestamp',freq='{}min'.format(t)),'Type']).Event_value.count()}).reset_index()
        df_gr5['Aggregate Ramp Passage flow rate'] = df_gr5['Aggregate Ramp Passage num vehicles'].apply(lambda x: x*(60/t))

        df_green_time_estimation = df_event_detail.loc[(df_event_detail['Type']=='Ramp Meter - Begin Green') | (df_event_detail['Type']=='Ramp Meter - Begin Red')].reset_index(drop=True)
        
        #df_gr3 is Green Time
        green_time_array = []
        all_start_timestamps = np.array(df_gr2['Timestamp'])
        for i in range(len(df_gr2)):
            s = df_gr2['Timestamp'][i]
            e = df_gr2['Timestamp'][i] + datetime.timedelta(minutes=t)
            df = df_green_time_estimation.loc[(df_green_time_estimation['Timestamp']>= df_gr2['Timestamp'][i])& (df_green_time_estimation['Timestamp']< df_gr2['Timestamp'][i]+datetime.timedelta(minutes=t))].reset_index(drop=True)
            if len(df)==0:
                green_time_array.append(0)
            else:
                green_time_array.append(green_time_calculator(df,s,e))
                
        df_gr3 = pd.DataFrame()
        df_gr3['Timestamp'] = all_start_timestamps
        df_gr3['Green_time'] = green_time_array
        #print(df_gr)
        #print(df_gr3)
        
        df_aggregate = pd.merge(df_gr, df_gr2[['Timestamp','Aggregate Mainline num vehicles','Aggregate Mainline flow rate']], on='Timestamp',how='left')
        df_aggregate = pd.merge(df_aggregate,df_gr3, on='Timestamp',how='left')
        df_aggregate = pd.merge(df_aggregate,df_gr4[['Timestamp','Aggregate Ramp Demand num vehicles','Aggregate Ramp Demand flow rate']], on='Timestamp',how='left')    
        df_aggregate = pd.merge(df_aggregate,df_gr5[['Timestamp','Aggregate Ramp Passage num vehicles','Aggregate Ramp Passage flow rate']], on='Timestamp',how='left') 
        return df_aggregate


    def preprocess_DeepSORT_data(df):
        df['enter_time'] = pd.to_datetime(df['enter_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        return df

    def aggregate_DeepSORT_data(df, t, lane_number):
        df_lane = df[df['lane']=='lane{}'.format(lane_number)].reset_index(drop=True)
        df_average_speed = pd.DataFrame({'Average Speed':df_lane.groupby(pd.Grouper(key='exit_time',freq='{}min'.format(t))).speed.mean()}).reset_index()
        df_max_speed = pd.DataFrame({'Max Speed':df_lane.groupby(pd.Grouper(key='exit_time',freq='{}min'.format(t))).speed.max()}).reset_index()
        
        df_average_flow = pd.DataFrame({'Number of Vehicles':df_lane.groupby(pd.Grouper(key='exit_time',freq='{}min'.format(t))).unique_ID.count()}).reset_index()
        df_average_flow['Average Flow Rate'] = df_average_flow['Number of Vehicles'].apply(lambda x: x*(60/t))
        df_aggregate = pd.merge(df_average_speed, df_average_flow,on='exit_time',how='left')
        df_aggregate = pd.merge(df_aggregate, df_max_speed, on='exit_time',how='left')
        df_aggregate['Average Vehicle Density'] = df_aggregate['Average Flow Rate']/df_aggregate['Average Speed']
        df_aggregate['Congestion measure'] = df_aggregate['Average Speed']/df_aggregate['Max Speed']
        df_aggregate = df_aggregate[(df_aggregate['exit_time']>=Timestamp(start_time))&(df_aggregate['exit_time']<=Timestamp(end_time))].reset_index(drop=True)
        column_names = ['Timestamp']
        all_columns = list(df_aggregate.columns)
        for name in range(1,len(all_columns)):
            column_names.append(all_columns[name]+ ' Lane {}'.format(lane_number)+' DeepSORT')
        df_aggregate.columns = column_names
        return df_aggregate

    
    def aggregate_all_lanes_DeepSORT(df, num_lanes, t):
        df_lane1 = preprocess_DeepSORT_data(df)
        df_lane1 = aggregate_DeepSORT_data(df_lane1, t, 1)
        df_lane2 = preprocess_DeepSORT_data(df)
        df_lane2 = aggregate_DeepSORT_data(df_lane2, t, 2)
        df_aggregate = pd.merge(df_lane1,df_lane2,on='Timestamp')
        
        for i in range(2, num_lanes):
            df = preprocess_DeepSORT_data(df)
            df = aggregate_DeepSORT_data(df,t,i+1)
            df_aggregate = pd.merge(df_aggregate,df,on='Timestamp',how='left')
        return df_aggregate


    def main(num_lanes, t):
        #Read loop detector data from database

        myoutputdb = pymysql.connect(host="localhost", \
            user="root", passwd="computerserver", \
            db="radishdb", port=3306)
        
        loop_data = 'select * from LoopDetectorData where ramp_id = %(ramp_id)s and enter_time between %(start_time)s and %(end_time)s;'
        loop_detector_data = pd.read_sql(loop_data, myoutputdb, params={'ramp_id': ramp_id, 'start_time': start_time, 'end_time': end_time})

        df_loop = preprocess_loop_detector_data(loop_detector_data)
        df_loop_aggregate = aggregate_loop_detector_data(df_loop,t)


        #Read Microwave data from database. Reading only Lane 1 data. Lane 1 is the lane adjacent to the ramp
        microwave_data = 'select * from MicrowaveData where ramp_id = %(ramp_id)s and enter_time between %(start_time)s and %(end_time)s;'
        microwave_detector_data = pd.read_sql(microwave_data, myoutputdb, params={'ramp_id': ramp_id, 'start_time': start_time, 'end_time': end_time})

        df_microwave = preprocess_microwave_detector_data(microwave_detector_data)
        df_microwave_aggregate = aggregate_microwave_detector_data(df_microwave,t,1)



        #Read Processed DeepSORT data from database
        tracks_data = 'select * from ProcessedTracks where ramp_id = %(ramp_id)s and enter_time between %(start_time)s and %(end_time)s;'
        deepsort_data = pd.read_sql(tracks_data, myoutputdb, params={'ramp_id': ramp_id, 'start_time': start_time, 'end_time': end_time})
        df_deepsort_aggregate = aggregate_all_lanes_DeepSORT(deepsort_data,num_lanes,t)

        #Aggregating data from all sensors
        df = pd.merge(df_loop_aggregate,df_microwave_aggregate,on='Timestamp',how='left')
        df = pd.merge(df,df_deepsort_aggregate,on='Timestamp',how='left')
        #df.to_csv('101 Cypress Results Data Aggregated over {} minutes interval.csv'.format(t),index=False)
        return df
    
    return main(1,t)


def plot_generator(ramp_id, start_time, end_time, aggregation_time, list_of_plots):
    df = aggregator(ramp_id, aggregation_time, start_time, end_time)

    n = len(list_of_plots)
    fig, axs = plt.subplots(n, 1, figsize=(15, 17))

    for index, value in enumerate(list_of_plots):
        if value == 'Aggregate Ramp Passage flow rate':
            axs[index].step(df['Timestamp'], df['Aggregate Ramp Passage flow rate'], color = 'red', where='post')
            axs[index].set_title('Aggregate Ramp Passage flow rate')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif value == 'Aggregate Ramp Demand flow rate':
            axs[index].step(df['Timestamp'], df['Aggregate Ramp Demand flow rate'], color = 'blue', where='post')
            axs[index].set_title('Aggregate Ramp Demand flow rate')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif value == 'Aggregate ramp metering rate':
            axs[index].step(df['Timestamp'], df['Aggregate ramp metering rate'], color = 'orange', where='post')
            axs[index].set_title('Aggregate ramp metering rate')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif value == 'Congestion measure Lane 1':
            axs[index].step(df['Timestamp'], df['Congestion measure Lane 1'], color = 'green', where='post')
            axs[index].set_title('Inverse Congestion measure Lane 1 Microwave')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif value == 'Congestion measure Lane 1 DeepSORT':
            axs[index].step(df['Timestamp'], df['Congestion measure Lane 1 DeepSORT'], color = 'salmon', where='post')
            axs[index].set_title('Inverse Congestion measure Lane 1 DeepSORT')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif value == 'Average Speed Lane 1':
            axs[index].step(df['Timestamp'], df['Average Speed Lane 1'], color = 'black', where='post')
            axs[index].set_title('Average Speed Lane 1 Microwave')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif value == 'Average Speed Lane 1 DeepSORT':
            axs[index].step(df['Timestamp'], df['Average Speed Lane 1 DeepSORT'], color = 'saddlebrown', where='post')
            axs[index].set_title('Average Speed Lane 1 DeepSORT')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif value == 'Average Occupancy Lane 1':
            axs[index].step(df['Timestamp'], df['Average Occupancy Lane 1'], color = 'darkcyan', where='post')
            axs[index].set_title('Average Occupancy Lane 1 Microwave')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif value == 'Average Vehicle Density Lane 1 DeepSORT':
            axs[index].step(df['Timestamp'], df['Average Vehicle Density Lane 1 DeepSORT'], color = 'violet', where='post')
            axs[index].set_title('Average Vehicle Density Lane 1 DeepSORT')
            axs[index].set_xlim(df['Timestamp'].min(), df['Timestamp'].max())
            axs[index].tick_params(axis='x', rotation=45)
            axs[index].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            axs[index].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.savefig('plot' + '.png')
    plt.clf()
    
    plt.close('all')
    


    
 


    
   